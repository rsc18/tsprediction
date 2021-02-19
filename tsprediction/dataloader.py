"""
DATALOADER
----------
Currently loads data from a csv file and provide a dataloader with given batch
size
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.preprocessing import MinMaxScaler
import random

# constants
SEQUENCE_LENGTH = 16


def get_input_and_target(dataset, sequence_length):
    """
    Takes timeseries data and sequence length as input and generates input 
    sequence and output sequence

    Parameters
    ----------
    dataset : numpy array
        time series data.
    sequence_length : int
        sequence length length for input and output sequence.

    Returns
    -------
    tuple
        A tuple of input samples and target samples

    """
    x = []
    y = []
    for i in range(len(dataset) - 2 * sequence_length + 1):
        _x = dataset[i : (i + sequence_length)]
        _y = dataset[i + sequence_length : i + sequence_length + sequence_length]
        x.append(_x)
        y.append(_y)
    return np.array(x), np.array(y)


def plot_from_csv(filename: str, plot_title: str):
    """
    Plots the close values of stock data from a csv file

    Parameters
    ----------
    filename : str
        The csv file name where the data is saved.
    plot_title : str
        The plot title.

    Returns
    -------
    None.

    """
    # loading the dataset from a csv file
    dataset = pd.read_csv(filename)
    dataset = dataset.iloc[:, 4].values

    # plotting the dataset
    plt.plot(dataset, label=plot_title)
    plt.show()


def dataloader_from_csv(
    filename: str,
    train_size_percentage: float = 0.7,
    train: bool = True,
    sequence_length: int = 64,
):
    """
    Loads the data from a csv file and returns train or test data 
    according to the splitting percentage.

    Parameters
    ----------
    filename : str
        The csv filename.
    train_size_percentage : float, optional
        train-test split. The default is 0.7.
    train : bool, optional
        train data or test data as returning tuple. The default is True.
    sequence_length : int, optional
        length of every sequence. The default is 64.

    Returns
    -------
    2D Tensor
        input sequence.
    2D Tensor
        target seuquence.

    """
    # loading the dataset from a csv file
    dataset = pd.read_csv(filename)
    dataset = dataset.iloc[:, 4].values
    dataset = dataset.reshape(-1, 1)
    # normalizing the data
    minMaxScalar = MinMaxScaler()
    normalized_data = minMaxScalar.fit_transform(dataset)

    # getting input and target sequence
    input_sequence, target_sequence = get_input_and_target(
        normalized_data, sequence_length
    )

    # train and test data size
    train_size = int(len(input_sequence) * train_size_percentage)
    test_size = len(input_sequence) - train_size

    dataX = Variable(torch.Tensor(np.array(input_sequence)))
    dataY = Variable(torch.Tensor(np.array(target_sequence)))

    data_all = [(x1, y1) for x1, y1 in zip(dataX, dataY)]

    random.shuffle(data_all)

    x_new = []
    y_new = []
    for x1, y1 in data_all:
        x_new.append(x1.numpy())
        y_new.append(y1.numpy())

    trainX = Variable(torch.Tensor(np.array(x_new[0:train_size])))
    trainY = Variable(torch.Tensor(np.array(y_new[0:train_size])))

    testX = Variable(torch.Tensor(np.array(x_new[train_size : len(x_new)])))
    testY = Variable(torch.Tensor(np.array(y_new[train_size : len(y_new)])))

    if train:
        return trainX, trainY
    else:
        return testX, testY


trX, trY = dataloader_from_csv(filename="microsoft.csv", train=True, sequence_length=64)
