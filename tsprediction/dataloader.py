"""
DATALOADER
----------
Currently loads data from a csv file and provide a dataloader with given batch
size
"""
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
from tsprediction.normalize_data import norm_data

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
    tuple list
        A tuple of input samples and target samples

    """
    input_seq = []
    target_seq = []
    for i in range(len(dataset) - 2 * sequence_length + 1):
        _x = dataset[i : (i + sequence_length)]
        _y = dataset[i + sequence_length : i + 2 * sequence_length]
        input_seq.append(_x)
        target_seq.append(_y)
    data_all = []
    for x_i, y_i in zip(np.array(input_seq), np.array(target_seq)):
        data_all.append((x_i, y_i))
    return data_all


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


def shuffled_input_target(dataset):
    """
    Returns shuffled input and target sequences
    Parameters
    ----------
    dataset : numpy array
        contains tuples of input and target samples

    Returns
    -------
    tuple
        A tuple of input samples and target samples

    """
    random.shuffle(dataset)
    input_sequence = []
    target_sequence = []
    for x_i, y_i in dataset:
        input_sequence.append(x_i)
        target_sequence.append(y_i)
    return input_sequence, target_sequence


def dataloader_from_csv(
    filename: str,
    train_size_percentage: float = 0.7,
    train: bool = True,
    sequence_length: int = 64,
):
    """
    Loads the data from a csv file and returns normalized train or test data
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
    dataset = ((pd.read_csv(filename)).iloc[:, 4].values).reshape(-1, 1)  ## why 4. close, check for seq length datalenght should be equal or greater than seq length
    # normalizing the data
    min_max_scalar = MinMaxScaler()
    normalized_data = min_max_scalar.fit_transform(dataset)

    # getting input and target sequence
    data_all = get_input_and_target(normalized_data, sequence_length)
    random.shuffle(data_all)
    input_sequence, target_sequence = shuffled_input_target(data_all)

    # train and test data size
    train_size = int(len(input_sequence) * train_size_percentage)

    train_x = Variable(torch.Tensor(np.array(input_sequence[0:train_size])))
    train_y = Variable(torch.Tensor(np.array(target_sequence[0:train_size])))

    test_x = Variable(
        torch.Tensor(np.array(input_sequence[train_size : len(input_sequence)]))
    )
    test_y = Variable(
        torch.Tensor(np.array(target_sequence[train_size : len(target_sequence)]))
    )

    if train:
        return train_x, train_y
    return test_x, test_y



def dataloader_from_pandas(
    df: pd.core.frame.DataFrame,
    train_size_percentage: float = 0.7,
    train: bool = True,
    sequence_length: int = 64,
    custom=False
):
    """
    Loads the data from a csv file and returns normalized train or test data
    according to the splitting percentage. 
    Parameters
    ----------
    df : pandas.core.frame.DataFrame,
         pandas dataFrame with time series data. We only extract the 5th column for now.
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
    if custom:
        dataset=df
    
    else:
        dataset = (df.iloc[:, 3].values).reshape(-1, 1)  ## why 4. close, check for seq length datalenght should be equal or greater than seq length
    
    # normalizing the data
    
    normalized_data=norm_data(dataset)
    # min_max_scalar = MinMaxScaler()
    # normalized_data = min_max_scalar.fit_transform(dataset)

    # getting input and target sequence
    data_all = get_input_and_target(normalized_data, sequence_length)
    random.shuffle(data_all)
    input_sequence, target_sequence = shuffled_input_target(data_all)

    # train and test data size
    train_size = int(len(input_sequence) * train_size_percentage)

    train_x = Variable(torch.Tensor(np.array(input_sequence[0:train_size])))
    train_y = Variable(torch.Tensor(np.array(target_sequence[0:train_size])))

    test_x = Variable(
        torch.Tensor(np.array(input_sequence[train_size : len(input_sequence)]))
    )
    test_y = Variable(
        torch.Tensor(np.array(target_sequence[train_size : len(target_sequence)]))
    )

    return train_x, train_y, test_x, test_y
