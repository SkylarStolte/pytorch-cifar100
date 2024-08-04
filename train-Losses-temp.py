# train.py
#!/usr/bin/env	python3

""" train network using pytorch

author baiyu
"""

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

from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_train_val_dataloaders

import pandas as pd
from DominoLoss_Multiply import DOMINO_Loss_Multiply
from DominoLoss import DOMINO_Loss
from hardl1ace import *

## Temperature scaling class
#class TemperatureScaling(nn.Module):
#    def __init__(self, net):
#        super(TemperatureScaling, self).__init__()
#        self.net = net
#        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

#    def forward(self, logits):
#        return logits / self.temperature

#def temperature_scaling(logits, labels):
#    net = TemperatureScaling(logits)
#    optimizer = optim.LBFGS([net.temperature], lr=0.01, max_iter=50)

#    def loss_fn():
#        loss = nn.CrossEntropyLoss()
#        return loss(net(logits), labels)
        
        #if args.loss_func=='CE':
        #    loss_function = nn.CrossEntropyLoss()
        #    loss = loss_function(outputs, labels)
        #elif args.loss_func=='DOMINO':
        #    loss_function = DOMINO_Loss()
        #    a = 0.8
        #    b = 0.3
        #    loss = loss_function(outputs, labels, matrix_penalty,a,b)
        #elif args.loss_func=='DOMINO_Multiply':
        #    loss_function = DOMINO_Loss_Multiply()
        #    loss = loss_function(outputs, labels, matrix_penalty,1)
        #elif args.loss_func=='ACE':   
        #    loss_function = HardL1ACEandCELoss(to_onehot_y=True)
        #    loss = loss_function(outputs, labels)

    #optimizer.step(loss_fn)
    #return net.temperature.item()

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):

        if args.gpu:
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        
        ######################################################################
        
        if args.loss_func=='CE' or args.loss_func=='ACE':
            loss = loss_function(outputs, labels)
        elif args.loss_func=='DOMINO':
            loss = loss_function(outputs, labels, matrix_penalty,a,b)
        elif args.loss_func=='DOMINO_Multiply':
            loss = loss_function(outputs, labels, matrix_penalty,1)
        
        ##loss = loss_function(outputs, labels)
        #loss = loss_function(outputs, labels, matrix_penalty,1) #, a, b)
        
        ######################################################################
        
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(cifar100_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        #print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        #    loss.item(),
        #    optimizer.param_groups[0]['lr'],
        #    epoch=epoch,
        #    trained_samples=batch_index * args.b + len(images),
        #    total_samples=len(cifar100_training_loader.dataset)
        #))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()

        outputs = net(images)
        ##loss = loss_function(outputs, labels)
        #loss = loss_function(outputs, labels, matrix_penalty, 1)#a, b)
        
        ######################################################################
        
        if args.loss_func=='CE' or args.loss_func=='ACE':
            loss = loss_function(outputs, labels)
        elif args.loss_func=='DOMINO':
            loss = loss_function(outputs, labels, matrix_penalty,a,b)
        elif args.loss_func=='DOMINO_Multiply':
            loss = loss_function(outputs, labels, matrix_penalty,1)
        
        ##loss = loss_function(outputs, labels)
        #loss = loss_function(outputs, labels, matrix_penalty,1) #, a, b)
        
        ######################################################################

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    #if args.gpu:
    #    print('GPU INFO.....')
    #    print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return correct.float() / len(cifar100_test_loader.dataset)

@torch.no_grad()
def eval_training_temp(epoch=0, tb=True):
    
    class TemperatureScaler(nn.Module):
        def __init__(self):
            super(TemperatureScaler, self).__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5, requires_grad=True)

        def forward(self, logits):
            return logits / self.temperature

        def set_temperature(self, val_loader, model):
            model.eval()
            nll_criterion = nn.CrossEntropyLoss().cuda()
            #nll_criterion.requires_grad = True
        
            logits_list = []
            labels_list = []
            #logits_list = torch.

            # Collect all logits and labels
            #counter = 0
            with torch.no_grad():
                for input, label in cifar100_validation_loader: #val_loader:
                    input, label = input.cuda(), label.cuda()
                    logits = model(input)
                    logits_list.append(logits)
                    labels_list.append(label)
                    
                    #if counter==0:
                    #    logits_all = logits.cuda()
                    #    labels_all = label.cuda()
                    #else:
                    #    logits_all = torch.cat([logits_all, logits]).cuda()
                        #logits.requires_grad=True
                        #labels = torch.cat(labels_list).cuda()
                    #    labels_all = torch.cat([labels_all, label]).cuda()
                        #labels.requires_grad=True
                    #counter += 1
                    
            logits = torch.cat(logits_list).cuda()
            labels = torch.cat(labels_list).cuda()

            # Reset requires_grad for temperature
            self.temperature.requires_grad = True

            # Optimizer for temperature scaling
            #optimizer = optim.Adam([self.temperature], lr=0.01)

            #for _ in range(500):  # Using a simple loop for optimization
            #    optimizer.zero_grad()
                #loss = nll_criterion(self.forward(logits_all), labels_all)
            #    logits_fin = logits/self.temperature
            #    logits_fin.requires_grad = True
            #    loss = nll_criterion(logits_fin, labels)
            #    loss.backward()
            #    optimizer.step()
            
            # Next: optimize the temperature w.r.t. NLL
            optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=500)

            def eval():
                optimizer.zero_grad()
                loss = nll_criterion(self.forward(logits), labels)
                loss.backward()
                return loss
            optimizer.step(eval)

            print(f'Optimal temperature: {self.temperature.item()}')

    start = time.time()
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0
    
    #outputs_list = []
    #labels_list = []
    #for (images, labels) in cifar100_test_loader:

    #    if args.gpu:
    #        images = images.cuda()
    #        labels = labels.cuda()
            
    #    outputs = net(images)
        
    #    outputs_list.append(outputs)
    #    labels_list.append(labels)
    #outputs = torch.cat(outputs_list)
    #labels = torch.cat(labels_list)

        ##loss = loss_function(outputs, labels)
        #loss = loss_function(outputs, labels, matrix_penalty, 1)#a, b)
        
    # Apply temperature scaling
    #optimal_temperature = temperature_scaling()#outputs, labels)
    #print(f"Optimal Temperature: {optimal_temperature}")

    # Calibrate the model
    #temperature_net = TemperatureScaling(net)
    #temperature_net.temperature = nn.Parameter(torch.ones(1) * optimal_temperature)
    
    temperature_scaler = TemperatureScaler().cuda()
    temperature_scaler.set_temperature(cifar100_validation_loader, net)
    
    def inference_with_temperature_scaling(input):
        logits = net(input)
        scaled_logits = temperature_scaler(logits)
        return scaled_logits
    
    net.eval()
    #temperature_net.eval()
    
    for (images, labels) in cifar100_test_loader:

        if args.gpu:
            images = images.cuda()
            labels = labels.cuda()
            

        outputs = inference_with_temperature_scaling(images)
            
        #outputs = temperature_net(net(images))
        
        ######################################################################
        
        if args.loss_func=='CE' or args.loss_func=='ACE':
            loss = loss_function(outputs, labels)
        elif args.loss_func=='DOMINO':
            loss = loss_function(outputs, labels, matrix_penalty,a,b)
        elif args.loss_func=='DOMINO_Multiply':
            loss = loss_function(outputs, labels, matrix_penalty,1)
        
        ##loss = loss_function(outputs, labels)
        #loss = loss_function(outputs, labels, matrix_penalty,1) #, a, b)
        
        ######################################################################

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    finish = time.time()
    #if args.gpu:
    #    print('GPU INFO.....')
    #    print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print('Test set: Epoch: {}, Average loss: {:.4f}, Accuracy: {:.4f}, Time consumed:{:.2f}s'.format(
        epoch,
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset),
        finish - start
    ))
    print()

    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(cifar100_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(cifar100_test_loader.dataset), epoch)

    return (correct.float() / len(cifar100_test_loader.dataset))#, temperature_net

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    parser.add_argument('-loss_func', type=str, default='CE', help='choose loss function')
    args = parser.parse_args()

    net = get_network(args)
    
    print(f"Epoch: {settings.EPOCH}")

    #data preprocessing:
    #cifar100_training_loader = get_training_dataloader(
    #    settings.CIFAR100_TRAIN_MEAN,
    #    settings.CIFAR100_TRAIN_STD,
    #    num_workers=4,
    #    batch_size=args.b,
    #    shuffle=True
    #)
    
    cifar100_training_loader, cifar100_validation_loader = get_train_val_dataloaders(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True
    )
    
    #####################################################################################
    
    matrix_dir = '/blue/ruogu.fang/skylastolte4444/Airplanes/Diffusion/'
    matrix_vals = pd.read_csv(matrix_dir + 'cifar100_matrix.csv', index_col=None, header=None) #'similarity.csv', index_col=0, header=0)#'cifar100_matrix.csv', index_col=None, header=None) 
    matrix_penalty = 3.0 * torch.from_numpy(matrix_vals.to_numpy())
    matrix_penalty = matrix_penalty.float().cuda()
    #print(matrix_penalty.shape)
    
    if args.loss_func=='CE':
        loss_function = nn.CrossEntropyLoss()
    elif args.loss_func=='DOMINO':
        loss_function = DOMINO_Loss()
        a = 0.8
        b = 0.3
    elif args.loss_func=='DOMINO_Multiply':
        loss_function = DOMINO_Loss_Multiply()
    elif args.loss_func=='ACE':   
        loss_function = HardL1ACEandCELoss(to_onehot_y=True)
    
    #####################################################################################

    #loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(cifar100_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 32, 32)
    if args.gpu:
        input_tensor = input_tensor.cuda()
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))


    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        
        if epoch==settings.EPOCH:
            acc = eval_training_temp(epoch)
        else:
            acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            
        #if epoch==settings.EPOCH:
        #    weights_path = checkpoint_path.format(net='Temp', epoch=epoch, type='regular')
        #    print('saving weights file to {}'.format(weights_path))
        #    torch.save(new_net.state_dict(), weights_path)

    writer.close()
