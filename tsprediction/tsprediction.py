"""
This module loads data from Dataloader module and  build LSTM model and we train it. 
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import random
from dataloader import dataloader_from_csv as dfc

seq_length=4
trainX,trainY=dfc('num.csv',train=True,sequence_length=seq_length)    
testX,testY=dfc('num.csv',train=False,sequence_length=seq_length)    


class LSTM(nn.Module):

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.seq_length = seq_length
        
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
    
num_epochs = 100
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
    if epoch % 20 == 0:
      print(outputs)
      print(trainY.view(trainY.shape[0], -1))
      print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
      
lstm.eval()
test_predict = lstm(testX[:50])
tp = test_predict.view(-1, 1)
data_predict = tp.data.numpy()
dataY = (testY[:50]).view(-1,1)
dataY_plot = dataY.data.numpy()
# data_predict = sc.inverse_transform(data_predict)
# dataY_plot = sc.inverse_transform(dataY_plot)
# plt.axvline(x=train_size, c='r', linestyle='--')

# plt.plot(dataY_plot)
# plt.plot(data_predict)
# plt.suptitle('Time-Series Prediction')
# plt.show()
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

