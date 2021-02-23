.. tspredictor documentation master file, created by
   sphinx-quickstart on Tue Feb  9 22:46:04 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to tspredictor's documentation!
=======================================
This repo contains the code base for predicting time series stock data. We will be using live stock data and feed them to the NN to train ehich will predict future stock values of given equity. We will be using LSTM for training.

Alpha- vantage: We will collect stock time series data using Alpha-vantage Time Series Stock API.
PyTorch: We will be using PyTorch library for building the Neural Network model with LSTM cells.
LSTM: Long Short Term Memory (LSTM) is an artificial recurrent neural network architecture which is capable of learning order dependence in sequence prediction problem.
Sequnce to Sequence Model: We will be building a sequence to sequence model with the LSTM cells. Here the input sequence would be a sequence of stock data of a specified equity and the output will be prediction of a future sequence of the same equity or some other equity.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
