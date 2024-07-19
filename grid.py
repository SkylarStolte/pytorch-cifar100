import itertools
import torch
import numpy as np
from sklearn.metrics import brier_score_loss

import os
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights

import pandas as pd
from DominoLoss import DOMINO_Loss

from torchmetrics.classification import MulticlassCalibrationError

def grid_search(train_loader, val_loader, net, loss_function, optimizer, matrix_penalty):
    best_a = None
    best_b = None
    best_ece = float('inf')

    a_values = np.linspace(.1,1,10)
    b_values = 1 - a_values #np.linspace(.1,1,10) #1 - a_values

    for a, b in zip(a_values, b_values):
        # Train the network with current a and b
        net.train()
        print("Starting Training")
        for batch_index, (images, labels) in enumerate(train_loader):
            #print("Starting Training")
            if args.gpu:
                labels = labels.cuda()
                images = images.cuda()
            torch.cuda.empty_cache()

            optimizer.zero_grad()
            outputs = net(images)
            del images
            loss = loss_function(outputs, labels, matrix_penalty, a, b)
            del outputs, labels
            torch.cuda.empty_cache()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

        # Evaluate on validation set
        print("Starting Evaluation")
        net.eval()
        #all_logits = []
        #all_labels = []
        ece = 0
        for images, labels in val_loader:
            if args.gpu:
                images = images.cuda()
                labels = labels.cuda()
            outputs = net(images)
            del images
            #all_logits.append(outputs)
            #all_labels.append(labels)
            torch.cuda.empty_cache()
            
            calibration_error = MulticlassCalibrationError(num_classes=outputs.size(1))
            ece += calibration_error(outputs,labels).item()
            
            torch.cuda.empty_cache()

        #all_logits = torch.cat(all_logits)
        #all_labels = torch.cat(all_labels)
        
        # Use MulticlassCalibrationError to compute ECE
        #calibration_error = MulticlassCalibrationError(num_classes=all_logits.size(1))
        #ece = calibration_error(all_logits, all_labels).item()
        torch.cuda.empty_cache()
        
        # Update best parameters
        if ece < best_ece:
            best_ece = ece
            best_a = a
            best_b = b
        print(f"Completed Run with a: {a} and b: {b}. ECE: {ece}")
        torch.cuda.empty_cache()
        del ece

    return best_a, best_b

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=64, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    # Initialize the network, optimizer, and loss function
    net = get_network(args)
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
    
    matrix_dir = '/blue/ruogu.fang/skylastolte4444/Airplanes/Diffusion/'
    #matrix_vals = pd.read_csv(matrix_dir + 'oxfordpets_ssim_matrix_norm4.csv', header = 0, index_col=0) #'Dictionary_matrixpenalty_inv_patches_v1_1024.csv', index_col = 0) #header=None
    #matrix_vals = pd.read_csv(matrix_dir + 'hc_matrixpenalty.csv', index_col = None, header=None)
    matrix_vals = pd.read_csv(matrix_dir + 'cifar100_matrix.csv', index_col=None, header=None) #'similarity.csv', index_col=0, header=0)#'cifar100_matrix.csv', index_col=None, header=None)
    matrix_penalty = 3.0 * torch.from_numpy(matrix_vals.to_numpy())
    matrix_penalty = matrix_penalty.float().cuda()
    print(matrix_penalty.shape)
    
    loss_function = DOMINO_Loss()

    # Perform grid search to find best a and b
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    val_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    best_a, best_b = grid_search(train_loader, val_loader, net, loss_function, optimizer, matrix_penalty)

    print(f"Best a: {best_a}, Best b: {best_b}")