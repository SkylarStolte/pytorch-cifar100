#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from conf import settings
from utils import get_network, get_test_dataloader

import torch.nn as nn
from torchmetrics.classification import MulticlassCalibrationError
from reliability_diagrams import *
import pandas as pd

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)

    cifar100_test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        #settings.CIFAR100_PATH,
        num_workers=4,
        batch_size=args.b,
    )

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(cifar100_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(cifar100_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')


            output = net(image)
            
            
            if n_iter==0:
                outputs_total = output.cpu().detach().numpy()
                #preds_total = outputs.argmax(dim=1)
                labels_total = label.cpu().detach().numpy()
            else:
                outputs_total = np.concatenate((outputs_total, output.cpu().detach().numpy()), axis=0)
                #preds_total = torch.cat((preds_total, outputs.argmax(dim=1)), dim=0)
                labels_total = np.concatenate((labels_total, label.cpu().detach().numpy()), axis=0)
                
            
            _, pred = output.topk(5, 1, largest=True, sorted=True)

            label = label.view(label.size(0), -1).expand_as(pred)
            correct = pred.eq(label).float()

            #compute top 5
            correct_5 += correct[:, :5].sum()

            #compute top1
            correct_1 += correct[:, :1].sum()

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    print()
    print("Top 1 err: ", 1 - correct_1 / len(cifar100_test_loader.dataset))
    print("Top 5 err: ", 1 - correct_5 / len(cifar100_test_loader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    
    #Calibration scores
    
    preds_total = torch.Tensor(outputs_total).argmax(dim=1)
    preds_total = preds_total.cpu().detach().numpy()
    
    m = nn.Softmax(dim=1)
    outputs_total = m(torch.Tensor(outputs_total))
    outputs_total = outputs_total.cpu().detach().numpy()

    o = torch.Tensor(outputs_total)
    l = torch.Tensor(labels_total)

    metric1 = MulticlassCalibrationError(num_classes=100, n_bins=10, norm='l1')
    ECE = metric1(o,l)
    metric2 = MulticlassCalibrationError(num_classes=100, n_bins=10, norm='l2')
    RMSCE = metric2(o,l)
    metric3 = MulticlassCalibrationError(num_classes=100, n_bins=10, norm='max')
    MCE = metric3(o,l)

    print('ECE: %.4f' % (ECE))
    print('RMSCE: %.4f' % (RMSCE))
    print('MCE: %.4f' % (MCE))

    #will need this to compute loss term
    data = [['ECE', ECE], ['RMSCE', RMSCE], ['MCE', MCE]]
    df_calmet = pd.DataFrame(data=data, columns=['Metric', 'Value'])
    df_calmet.to_csv('calibrationmetrics.csv')
    
    #Reliability Diagrams
    
    #plt.style.use("seaborn")

    plt.rc("font", size=12)
    plt.rc("axes", labelsize=12)
    plt.rc("xtick", labelsize=12)
    plt.rc("ytick", labelsize=12)
    plt.rc("legend", fontsize=12)

    plt.rc("axes", titlesize=16)
    plt.rc("figure", titlesize=16)
    title = "Total Calibration Curve"

    output_conf = np.max(outputs_total, axis=1)

    fig = reliability_diagram(labels_total, preds_total, output_conf, num_bins=10, draw_ece=True, draw_bin_importance="alpha", draw_averages=True, title=title, figsize=(6, 6), dpi=100, return_fig=True)

    fig.tight_layout()

    fig.savefig('allclass_calibrationcurve' + '.pdf')
    plt.close()

    def plot_confusion_matrix(labels, pred_labels, classes):

        fig = plt.figure(figsize=(20, 20))
        ax = fig.add_subplot(1, 1, 1)
        cm = confusion_matrix(labels, pred_labels)
        cm = ConfusionMatrixDisplay(cm, display_labels=classes)
        cm.plot(values_format='d', cmap='Blues', ax=ax)
        plt.grid(False)
        plt.xticks(rotation=90)

    #class_names = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]
    
    cifar100_classes = ["Apple", "Aquarium fish", "Baby", "Bear", "Beaver", "Bed", "Bee", "Beetle", "Bicycle", "Bottle", "Bowl", "Boy", "Bridge", "Bus", "Butterfly", "Camel", "Can", "Castle", "Caterpillar", "Cattle", "Chair", "Chimpanzee", "Clock", "Cloud", "Cockroach", "Couch", "Crab", "Crockpot", "Cup", "Dinosaur", "Dolphin", "Elephant", "Flatfish", "Forest", "Fox", "Girl", "Hamster", "House", "Kangaroo", "Keyboard", "Lamp", "Lawn mower", "Leopard", "Lion", "Lizard", "Lobster", "Man", "Maple tree", "Motorcycle", "Mountain", "Mouse", "Mushroom", "Oak tree", "Orange", "Orchid", "Otter", "Palm tree", "Pear", "Pickup truck", "Pine tree", "Plain", "Plate", "Poppy", "Porcupine", "Possum", "Rabbit", "Raccoon", "Ray", "Road", "Rocket", "Rose", "Sea", "Seal", "Shark", "Shrew", "Skunk", "Skyscraper", "Snail", "Snake", "Spider", "Squirrel", "Streetcar", "Sunflower", "Sweet pepper", "Table", "Tank", "Telephone", "Television", "Tiger", "Tractor", "Train", "Trout", "Tulip", "Turtle", "Wardrobe", "Whale", "Willow tree", "Wolf", "Woman", "Worm"]
    
    plot_confusion_matrix(labels_total, preds_total, class_names)
    plt.tight_layout()
    plt.savefig('confusionmatrix_test.png')

    #will need this to compute loss term
    df_cm = pd.DataFrame(confusion_matrix(labels_total,preds_total))
    df_cm.to_csv('confusionmatrix_test.csv')