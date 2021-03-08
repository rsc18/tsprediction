# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 19:54:10 2021

@author: rsc18
"""
import torch
import torch.nn as nn

#import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import sys
import os

CURRENT_DIR = os.getcwd()
PARENT_DIR = "/".join(CURRENT_DIR.split("/")[:-1])
sys.path.append(PARENT_DIR)

import tsprediction.dataloader as dl
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
'''
from alpha_vantage.timeseries import TimeSeries

ts = TimeSeries(key='2JVYQIAGV1ABJD0O', output_format='pandas')
data_ms, meta_data_ms = ts.get_intraday(symbol='TSLA',interval='1min', outputsize='full')
# data_ms, meta_data_ms = ts.get_monthly(symbol='TSLA')

training_set = data_ms.iloc[:,3:4]['4. close'].values.astype(float)
'''

all_data = torch.rand((100,))

train_data_percentage = 0.8

total_length = len(all_data)
divison  = int(total_length*train_data_percentage)

train_data = all_data[:divison]
test_data = all_data[divison:]


scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

train_loader = dl.load(train_data_normalized, batch_size = 1, shuffle = True)
dataiter = iter(train_loader)

x,y = dataiter.next()
'''
test_data_size =75

train_data = all_data[:-test_data_size]
test_data = all_data[-test_data_size:]

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data .reshape(-1, 1))

train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1).to(device)


train_window = 365

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
'''
#train_inout_seq = create_inout_sequences(train_data_normalized, train_window)


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=100, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size)

        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                            torch.zeros(1,1,self.hidden_layer_size))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]
    
model = LSTM().to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200

for i in range(epochs):
    for seq, labels in train_loader:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size).to(device),
                        torch.zeros(1, 1, model.hidden_layer_size).to(device))

        y_pred = model(seq.to(device))

        single_loss = loss_function(y_pred, labels.to(device))
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred =75

test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:]).to(device)
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        
        test_inputs.append(model(seq).item())
 
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:] ).reshape(-1, 1))

# x = np.arange(132, 144, 1)

x=list(data_ms.index)[0:75]

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

# plt.plot(data_ms['4. close'][-train_window:])
plt.plot(data_ms['4. close'])
plt.plot(x,actual_predictions)
plt.show()
