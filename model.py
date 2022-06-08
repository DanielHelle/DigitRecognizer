import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch import optim

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,22,kernel_size=5)
        self.conv2 = nn.Conv2d(22, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32,32, kernel_size=5)
        self.conv4 = nn.Conv2d(32,64, kernel_size=5)

        
        
        
    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(x,kernel_size=2))
        x = F.relu(self.conv3(x))
        x = F.relu(self.con4(x))
        x = F.max_pool2d(x,kernel_size=2)
        print(x.size())
        return x
        
        
       # x = x.view(-1,)
        
        


def fit(model,train_loader):
    optimizer = optim.Adam(model.parameters())
    loss = nn.CrossEntropyLoss()
    epochs = 6
    batch = 30
    model.train()
    for epoch in range(epochs):
        correct = 0
        for i, data in enumerate(train_loader):
            X, y = data
            print(X.size())
            optimizer.zero_grad()
            pred = model(X)
            loss = loss(pred,y)
            loss.backward()
            optimizer.step()
            correct += torch.sum(pred == y)
            if i % 30 == 0:
                print("Epoch: {}, Correct: {}, Batch {} ".format(epoch, correct/batch,i))