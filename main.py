"""
Code to use the saved models for testing
"""

import numpy as np
import pdb
import os
from tqdm import tqdm

from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision.datasets import FashionMNIST
from torchvision import transforms

#from utils import AverageMeter

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class classify3(nn.Module):
  def __init__(self):
    super(classify3,self).__init__()
    self.conv1 = nn.Conv2d(1,32,3)
    self.mx = nn.MaxPool2d(2,2)
    self.fc1 = nn.Linear(5408,100)
    self.fc2 = nn.Linear(100,10)

  def forward(self,x):
    x = (self.mx(F.relu(self.conv1(x))))
    x = torch.flatten(x,start_dim = 1)
    return self.fc2(F.relu(self.fc1(x)))

class classify(nn.Module):
  def __init__(self):
    super(classify,self).__init__()
    self.fc1 = nn.Linear(784,200)
    self.fc2 = nn.Linear(200,100)
    self.fc3 = nn.Linear(100,10)
  #Vishnu
  def forward(self,x):
    x = self.fc3(F.relu(self.fc2(F.relu(self.fc1(x)))))
    return x

net = classify()
net3 = classify3()
criterion = nn.CrossEntropyLoss()

def test(model, testloader, model_type):
    """ Training the model using the given dataloader for 1 epoch.

    Input: Model, Dataset, optimizer,
    """

    model.eval()

    y_gt = []
    y_pred_label = []

    correct1 = 0
    correct = 0
    total1 = 0
    total = 0
    i = 0
    r_l = 0
    for data in testloader:
        input_s,label_s = data
        if model_type == "nn":
            input_s = torch.flatten(input_s,start_dim=1)

        output_s1 = model(input_s)

        loss = criterion(output_s1, label_s)
        r_l += loss.item()
        i = i+1

        _,predicted1 = torch.max(output_s1.data,1)
        y_pred_label += list(predicted1.numpy())
        y_gt += list(label_s.numpy())
        total += label_s.size(0)

        correct1 += (predicted1 == label_s).sum().item()

    loss_v = r_l/(i+1)
    accuracy = correct1/total

    #for batch_idx, (img, y_true) in enumerate(testloader):
        #img = Variable(img)
        #y_true = Variable(y_true)
        #out = model(img)
        #y_pred = F.softmax(out, dim=1)
        #y_pred_label_tmp = torch.argmax(y_pred, dim=1)

        #loss = F.cross_entropy(out, y_true)
        #avg_loss.update(loss, img.shape[0])

        # Add the labels
        #y_gt += list(y_true.numpy())
        #y_pred_label += list(y_pred_label_tmp.numpy())

    return loss_v, accuracy, y_gt, y_pred_label


if __name__ == "__main__":

    test_s = torchvision.datasets.FashionMNIST(root='./data/FashionMNIST',train=False,download=True,transform = transforms.Compose([transforms.ToTensor()]))
    testloader = torch.utils.data.DataLoader(test_s,batch_size=4,shuffle=False)

    # Load the models
    path1 = 'model/net_34.pth'
    path2 = 'model/net_334.pth'

    net.load_state_dict(torch.load(path1))
    net3.load_state_dict(torch.load(path2))
    
    loss, accuracy, gt, pred = test(net, testloader, 'nn')
    with open("multi-layer-net.txt", 'w') as f:
        f.write("Loss on Test Data : \n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(accuracy))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))

    loss, accuracy,gt, pred = test(net3, testloader,'cnn')
    with open("convolution-neural-net.txt", 'w') as f:
        f.write("Loss on Test Data : {}\n".format(loss))
        f.write("Accuracy on Test Data : {}\n".format(accuracy))
        f.write("gt_label,pred_label \n")
        for idx in range(len(gt)):
            f.write("{},{}\n".format(gt[idx], pred[idx]))
