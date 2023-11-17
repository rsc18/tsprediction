# Sequence to Sequence Stock Time Series Data Prediction Model using PyTorch 

#### Project Status: [Active, On-Hold, Completed]
Completed

## Project Intro/Objective
This repo contains the code base for predicting time series stock data. We will be using live stock data and feed them to the NN to train ehich will predict future stock values of given equity. We will be using LSTM for training. 

## Project Description
* Alpha- vantage: We will collect stock time series data using Alpha-vantage Time Series Stock API. 
* PyTorch: We will be using PyTorch library for building the Neural Network model with LSTM cells.
* LSTM: Long Short Term Memory (LSTM) is an artificial recurrent neural network architecture which is capable of learning order dependence in sequence prediction problem.
* Sequnce to Sequence Model: We will be building a sequence to sequence model with the LSTM cells. Here the input sequence would be a sequence of stock data of a specified equity and the output will be prediction of a future sequence of the same equity or some other equity. 

## Installation   
 * git clone https://github.com/rsc18/tsprediction.git
 * Run "poetry install" from root directory to install all required libraries/home/ram/.cache/pypoetry/virtualenvs/tsprediction-Yr_4QMnV-py3.8
 * Run "poetry shell" to activate the virtual env
 * To run the project follow the Usage
 
## Usage
 * Create Alphavantage key from https://www.alphavantage.co/support/#api-key
 * Save that key for later use

```
    Predict Stock CLI

    Usage:
    predict_stock.py alphavantage <alphavantage_key> <companySymbol> <predictSequenceLength> [options]
    predict_stock.py custom <csvFileLocation> <predictSequenceLength> [options]
    predict_stock.py --help
    predict_stock.py --listcompanies -k <keywords>

    Options:
    -h --help                   help with predict_stock useage
    --plot= True/False          plot flag if true saves the plots in plot folder
    --saveModel= model-name     save_model flag if given saves the trained model with
                                given model-name for future use
    -k <keywords>               search companies names using keyword
    --epochs = no-of-epochs     no of epochs
    --category = stock category open, high, low, close, volume
    --saveCSV = True/False      save the predicted data in CSV file

```

## Example
To list companies symbol that match keyword "micro" :

``` python predict_stock.py --listcompanies -k micro  ``` 
     
To train a stock model with MSFT(Microsoft) data. (Note the alphavantage key might not work for you create your own from https://www.alphavantage.co/support/#api-key)
    
``` python predict_stock.py alphavantage B4CVMOCB6M2W7B5O MSFT 64```     
 
![alt text](https://github.com/rsc18/tsprediction/blob/main/figures/MSFT-64-e300.png)

To train a model with custom datasets.  We have a sawtooth data in num.csv.

``` python predict_stock.py custom tsprediction/data/num.csv 9   ```     
  
![alt text](https://github.com/rsc18/tsprediction/blob/main/figures/sawtooth.png)   

Results from Suchita's Data:
We used apple's data.

``` python predict_stock.py custom tsprediction/data/aapl.csv 32   ``` 
    
![alt text](https://github.com/rsc18/tsprediction/blob/main/figures/S-AAPL-32-e300.png)

## Running Pylint and Coverage   
Run following command from root folder    
make pylint   
make coverage   

## Generating docs
Go to documentation folder and run : make html

### Technologies
* Python
* Pandas
* Spyder
* PyTorch
 
 
=======
### From
|Name     | e-mail  |
|---------|-------|
| Ram Sharan Chaulagain |  rsc18@my.fsu.edu |
| Chashi Mahiul Islam | ci20l@my.fsu.edu |
