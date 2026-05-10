# Authors: xxx 
# Copyright (c) 2023, The Regents of the xxx 
# All rights reserved.

import torch
import mlflow, mlflow.pytorch
import numpy as np
import os
from model import UNET, FCN
from data import ImageDataset, collate_fn
from torch.utils.data import Dataset, DataLoader
import pickle as pkl
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
# from sklearn.metrics import f1_score
from datetime import datetime
import sys
import matplotlib.pyplot as plt

def f1_score(y_true, y_pred):
    tp = np.sum(y_true * y_pred).astype(np.float32)
    tn = np.sum((1 - y_true) * (1 - y_pred)).astype(np.float32)
    fp = np.sum((1 - y_true) * y_pred).astype(np.float32)
    fn = np.sum(y_true * (1 - y_pred)).astype(np.float32)
    
    epsilon = 1e-7

    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2 * (precision * recall) / (precision + recall + epsilon)
    return f1, accuracy, tp, tn, fp, fn, precision, recall

def evaluate(model, criterion, dataloader, device):
    model.eval()  # Set the model to evaluation mode
    f1_scores = []
    accuracies = []
    tps = []
    tns = []
    fps = []
    fns = []
    precisions = []
    recalls = []
    losses = []
    with torch.no_grad():  # No gradients required during evaluation
        for images, masks in dataloader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks.float())
            losses.append(loss.item())
            # Use threshold of 0.5 to binarize predictions
            preds = (outputs > 0.0).float()

            ## Convert preds to numpy
            preds = preds.cpu().numpy()
            masks = masks.cpu().numpy()
            ## Convert masks and preds from cuda to np
            f_score, accuracy, tp, tn, fp, fn, precision, recall = f1_score(masks, preds)
            f1_scores.append(f_score)
            accuracies.append(accuracy)
            tps.append(tp)
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
            precisions.append(precision)
            recalls.append(recall)
    
    # Mean of f1_scores without using np.mean as f1_scores elements are Torch.tensor
    f_score = sum(f1_scores)/len(f1_scores)
    accu = sum(accuracies)/len(accuracies)
    tp = sum(tps)/len(tps)
    tn = sum(tns)/len(tns)
    fp = sum(fps)/len(fps)
    fn = sum(fns)/len(fns)
    precision = sum(precisions)/len(precisions)
    recall = sum(recalls)/len(recalls)
    loss = np.mean(losses)
    return f_score, accu, loss, tp, tn, fp, fn, precision, recall

def train_one_epoch(train_loader, model, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    f_scores = []
    accuracies = []
    tps = []
    tns = []
    fps = []
    fns = []
    precisions = []
    recalls = []
    loss = None
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        ## Comput accuracy ##
        loss = criterion(output, target.float())
        epoch_loss += loss.item()

        preds = (output > 0.0).float()
        preds = preds.cpu().numpy()
        target = target.cpu().numpy()
        f_score, accuracy, tp, tn, fp, fn, precision, recall  = f1_score(target, preds)
        f_scores.append(f_score)
        accuracies.append(accuracy)
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
        precisions.append(precision)
        recalls.append(recall)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    tp = sum(tps)/len(tps)
    tn = sum(tns)/len(tns)
    fp = sum(fps)/len(fps)
    fn = sum(fns)/len(fns)
    precision = sum(precisions)/len(precisions)
    recall = sum(recalls)/len(recalls)

    return np.mean(f_scores), np.mean(accuracies), epoch_loss/len(train_loader), tp, tn, fp, fn, precision, recall

def train(train_data:str, test_data:str, device:int, epochs:int = 300,
          batch_size:int = 2, is_box:bool = True, is_unet:bool = True) -> None:
    
    # Get current date and time
    print(f"Is_box: {is_box}, Is_unet: {is_unet}")
    now = datetime.now()
    # Format as a string: YYYYMMDD_HHMMSS
    prefix = now.strftime("%Y%m%d_%H%M%S")
    if is_unet:
        prefix += "_unet"
    else:
        prefix += "_routenet"
    
    if is_box:
        prefix += "_box"
    else:
        prefix += "_mask"
    
    # prefix += "_200"
    
    train_dataset = ImageDataset(train_data, True, is_box)
    print(f"Number of train data: {len(train_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers = 2, collate_fn=collate_fn)
    
    test_dataset = ImageDataset(test_data, is_box=is_box)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers = 2, collate_fn=collate_fn)
    
    device = torch.device(f'cuda:{device}' if torch.cuda.is_available() else 'cpu')
    ## Print Data shape for train and test ##
    print(f"Train data shape: {train_dataset[0][0].shape} {train_dataset[0][1].shape}")
    print(f"Test data shape: {test_dataset[0][0].shape} {test_dataset[0][1].shape}")
    model = UNET()
    lr = 0.00001
    
    if not is_unet:
        print("Using FCN")
        model = FCN()
        lr = 0.00001
    
    model = model.to(device)
    # data_weight = torch.tensor([0.6/0.4])
    data_weight = torch.tensor([10.0])
    
    criterion = nn.BCEWithLogitsLoss(pos_weight = data_weight)
    criterion = criterion.to(device)
    # For UNET best lr = 0.0001
    # For FCN best lr = 0.001
    # lr = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr,
    #                                                 steps_per_epoch=len(train_loader),
    #                                                 epochs = epochs)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                           factor=0.5)
    best_test_loss = float('-inf')
    best_model_path = f"./best_model/ISPD_pre/best_model_{prefix}.pth"

    mlflow.set_experiment(f"Tomography_ISPD_pre_shuffle")
    
    with mlflow.start_run(run_name = prefix):
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epoch", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("weight", data_weight)
        for epoch in range(1, epochs+1):
            train_f1, train_acc, train_loss, train_tp, train_tn, train_fp, train_fn, train_prec, train_recall = train_one_epoch(train_loader,
                                                                model, criterion,
                                                                optimizer, device)
            test_f1, test_acc, test_loss, test_tp, test_tn, test_fp, test_fn, test_prec, test_recall = evaluate(model, criterion, test_loader, device)

            mlflow.log_metric("train_f1", train_f1, step = epoch)
            mlflow.log_metric("train_acc", train_acc, step = epoch)
            mlflow.log_metric("train_loss", train_loss, step = epoch)
            mlflow.log_metric("train_tp", train_tp, step = epoch)
            mlflow.log_metric("train_tn", train_tn, step = epoch)
            mlflow.log_metric("train_fp", train_fp, step = epoch)
            mlflow.log_metric("train_fn", train_fn, step = epoch)
            mlflow.log_metric("train_prec", train_prec, step = epoch)
            mlflow.log_metric("train_recall", train_recall, step = epoch)
            mlflow.log_metric("test_f1", test_f1, step = epoch)
            mlflow.log_metric("test_acc", test_acc, step = epoch)
            mlflow.log_metric("test_loss", test_loss, step = epoch)
            mlflow.log_metric("test_tp", test_tp, step = epoch)
            mlflow.log_metric("test_tn", test_tn, step = epoch)
            mlflow.log_metric("test_fp", test_fp, step = epoch)
            mlflow.log_metric("test_fn", test_fn, step = epoch)
            mlflow.log_metric("test_prec", test_prec, step = epoch)
            mlflow.log_metric("test_recall", test_recall, step = epoch)

            if scheduler is not None:
                scheduler.step(test_loss)
            print(f"Epoch: {epoch}, Train loss: {train_loss}, Test loss:"
                f" {test_loss} Train F1 Score: "
                f"{train_f1}, Test F1 Score: {test_f1}, Train acc: "
                f"{train_acc}, Test acc: {test_acc}, Train tp: "
                f"{train_tp}, Test tp: {test_tp}, Train fp: "
                f"{train_fp}, Test fp: {test_fp}, Train tn: "
                f"{train_tn}, Test tn: {test_tn}, Train fn: "
                f"{train_fn}, Test fn: {test_fn}, Train prec: "
                f"{train_prec}, Test prec: {test_prec}, Train recall: "
                f"{train_recall}, Test recall: {test_recall}")
            if test_f1 > best_test_loss:
                best_test_loss = test_f1
                torch.save(model.state_dict(), best_model_path)
                print(f"Saved model at epoch {epoch} with Test loss: {test_loss} "
                    f"Test F1 Score: {test_f1}")
    
    print(f"Best model path: {best_model_path}")

    mlflow.end_run()
    
if __name__ == '__main__':
    train_data = '/home/frankfu/Tomography/util/data/step1/ISPD_2015_no_route/train/data.txt'
    test_data = '/home/frankfu/Tomography/util/data/step1/ISPD_2015_no_route/test/data.txt'
    
    device = int(sys.argv[1])
    is_unet = bool(int(sys.argv[2]))
    is_box = bool(int(sys.argv[3]))
    train(train_data, test_data, device, is_unet=is_unet, is_box=is_box)
