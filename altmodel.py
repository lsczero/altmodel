#!/usr/bin/env
# -*- coding: UTF-8 -*-
# Copyright (C) @2023 Shaochuang Liu. All right Reserved.
# @Author:wnight
# @FileName:train.py
# @DateTime:2023/9/6 9:59
# @SoftWare: PyCharm
# You are not expected to understand this
import tables
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.functional import F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import Dataset, DataLoader
import logging
import os
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet1D(nn.Module):
    def __init__(self, block, layers, input_channels=1, num_classes=10):
        super(ResNet1D, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class NanoporeDataset(Dataset):
    def __init__(self,datas,labels):
        self.x = datas
        self.y = labels

    def __getitem__(self,index):
        return self.x[index, :].astype("float32").reshape(1,10000), self.y[index].astype("int64")

    def __len__(self):
        return len(self.y)

def loaddatasets(fn):
    with tables.open_file(fn,"r",) as hf:
        datas = hf.root["datas"]["data"].read()
        labels = hf.root["datas"]["label"].read()
    npdata = NanoporeDataset(datas, labels)
    return npdata

def creatresnet(layer=18,input_channels=1,num_classes=2):

    resnet18= [2, 2, 2, 2]  # resnet18
    resnet34= [3, 4, 6, 3] # resnet18
    if layer==18:
        layers = resnet18
    elif layer==34:
        layers = resnet34
    return ResNet1D(ResidualBlock1D, layers, input_channels=input_channels, num_classes=num_classes)

def seed_torch(seed=2023):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    fn = "padres.hdf5"
    seed_torch(2023)
    epochs = 1000
    batchsize = 200
    lr = 0.001  # 学习率
    labels = ['A11',
             'A3CA7',
             'A3GA7',
             'A3TA7',
             'A5CA5',
             'A5GA5',
             'A5TA5',
             'A7CA3',
             'A7GA3',
             'A7TA3']

    npdata = loaddatasets(fn)
    train_dataset, test_dataset = torch.utils.data.random_split(npdata,[18000, 4638])
    train_loader = DataLoader(dataset=train_dataset, batch_size=batchsize,
                              shuffle=True, num_workers=5,pin_memory=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batchsize,
                             shuffle=True, num_workers=5)
    model = creatresnet(34,1,10)
    criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr,betas=(0.9, 0.99))
    optimizer = torch.optim.SGD(model.parameters(),lr=lr)
    device = torch.device("cuda:0")

    writer = SummaryWriter(log_dir="runlog")

    model.train()
    for epoch in range(epochs):
        train_loss = 0
        correct = 0
        loss_mean = 0
        total = 0
        # training
        for index, data in enumerate(train_loader):
            optimizer.zero_grad()
            inputs, labels = data
            inputs = inputs.to(device)
            label = labels.to(device)
            output = model(inputs)
            _, pred = torch.max(output.data, 1)
            loss = criterion(output, label)
            train_loss += loss
            loss.backward()
            optimizer.step()

            total += label.size(0)
            correct += (pred==label).sum().item()

        loss_mean = train_loss / (index + 1)
        writer.add_scalar("loss/train", loss_mean, epoch)
        writer.add_scalar("accuracy/train", correct/total, epoch)
        print(f"{epoch}, {loss_mean}")
        print('train_Accuracy: %.2f %%' % (100 * correct / total))
        if epoch % 200 == 0:
            torch.save(model.state_dict(), f"runlog/resnetnovmd_epoch{epoch}.pth")
        torch.save(model.state_dict(), f"runlog/resnetnovmd_last.pth")
    # evaluate
    model.eval()
    correct = 0
    total = 0
    real_label = []
    pred_label = []
    class_probs = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            label = labels.to(device)
            output = model(inputs)
            _, pred = torch.max(output.data, 1)
            total += label.size(0)
            correct += (pred == label).sum().item()
            real_label.append(label.cpu())
            pred_label.append(pred.cpu())
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            class_probs.append(class_probs_batch)

        writer.add_scalar("Accuracy_last", correct / total, i + 1)
        print(f'test_Accuracy of last: {correct / total:.2%}')

        real_label = np.concatenate(real_label)
        pred_label = np.concatenate(pred_label)
        res = np.column_stack((real_label, pred_label))
        C2 = confusion_matrix(real_label, pred_label, labels=list(range(10)))
        precision, recall, thresholds = precision_recall_curve(real_label, pred_label)
        print(precision, recall, thresholds)
        test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
        np.save("result_confusion_matrix.npy", C2)
        np.save("result_lasteval.npy", res)