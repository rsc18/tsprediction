"""
Dataloader for Alphga-Vantage
-----------------------------
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import sys
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# from alpha_vantage.timeseries import TimeSeries
'''
CURRENT_DIR = os.getcwd()
PARENT_DIR = "/".join(CURRENT_DIR.split("/")[:-1])
sys.path.append(PARENT_DIR)
import datasets.raw_data_extractor as rde

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = rde.get_intraday_dataset(symbol, interval, key)
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

batch_size = 16
window = int(divison/16)



def create_inout_sequences(input_data, tw):
    x = []
    y = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        x.append(np.array(train_seq))
        train_label = input_data[i+tw:i+tw+1]
        y.append(np.array(train_label))
    tensor_x = torch.Tensor(x)
    tensor_y = torch.Tensor(y)
    dataset = TensorDataset(tensor_x, tensor_y)
    return dataset


def load(dataset, batch_size=16, shuffle=True):
    window = int(divison/batch_size)
    dataset = create_inout_sequences(dataset, window)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

train_loader = load(train_data_normalized, batch_size = 16, shuffle = True)

dataiter = iter(train_loader)

x,y = dataiter.next()
