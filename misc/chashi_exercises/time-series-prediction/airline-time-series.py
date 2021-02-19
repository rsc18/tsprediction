#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 17:57:14 2021

@author: chashi
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import random
training_set = pd.read_csv('microsoft.csv')
#training_set = pd.read_csv('shampoo.csv')

training_set = training_set.iloc[1:,4:5].values

#plt.plot(training_set, label = 'Shampoo Sales Data')
plt.plot(training_set, label = 'Microsoft Stock Data')
plt.show()

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-2*seq_length + 1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length : i+seq_length + seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)

sc = MinMaxScaler()
training_data = sc.fit_transform(training_set)

training_data = np.array([i for i in range(81) if i%2!=0])
training_data = training_data.reshape(20,1)
seq_length = 4
x, y = sliding_windows(training_data, seq_length)

train_size = int(len(y) * 0.67)
test_size = len(y) - train_size

dataX = Variable(torch.Tensor(np.array(x)))
dataY = Variable(torch.Tensor(np.array(y)))

data_all = [(x1, y1) for x1,  y1 in zip(dataX, dataY)]

random.shuffle(data_all)

x_new = []
y_new = []
for x1,y1 in data_all:
    x_new.append(x1.numpy())
    y_new.append(y1.numpy())
    
trainX = Variable(torch.Tensor(np.array(x_new[0:train_size])))
trainY = Variable(torch.Tensor(np.array(y_new[0:train_size])))

testX = Variable(torch.Tensor(np.array(x_new[train_size:len(x)])))
testY = Variable(torch.Tensor(np.array(y_new[train_size:len(y)])))

class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size))
        
        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        
        h_out = h_out.view(-1, self.hidden_size)
        
        out = self.fc(h_out)
        
        return out
    
num_epochs = 10000
learning_rate = 0.01

input_size = 1
hidden_size = 8
num_layers = 1

num_classes = seq_length

lstm = LSTM(num_classes, input_size, hidden_size, num_layers)

criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
#optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
# Train the model
#o = lstm(trainX)
for epoch in range(num_epochs):
    outputs = lstm(trainX)
    optimizer.zero_grad()
    # obtain the loss function
    loss = criterion(outputs, trainY.view(trainY.shape[0], -1))
    
    loss.backward()
    
    optimizer.step()
    if epoch % 200 == 0:
      print(outputs)
      print(trainY.view(trainY.shape[0], -1))
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
      
lstm.eval()
test_predict = lstm(testX)
tp = test_predict.view(-1, 1)
data_predict = tp.data.numpy()
dataY = testY.view(-1,1)
dataY_plot = dataY.data.numpy()
# =============================================================================
# 
# data_predict = sc.inverse_transform(data_predict)
# dataY_plot = sc.inverse_transform(dataY_plot)
# =============================================================================

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()
'''
train_predict = lstm(dataX)
tp = train_predict.view(-1, 1)
data_predict = tp.data.numpy()
dataY = dataY.view(-1,1)
dataY_plot = dataY.data.numpy()

data_predict = sc.inverse_transform(data_predict)
dataY_plot = sc.inverse_transform(dataY_plot)

plt.axvline(x=train_size, c='r', linestyle='--')

plt.plot(dataY_plot)
plt.plot(data_predict)
plt.suptitle('Time-Series Prediction')
plt.show()
'''
