"""Predict Stock CLI

Usage:
    predict_stock.py alphavantage <companySymbol> <predictSequenceLength> [options]
    predict_stock.py custom <csvFileLocation> <predictSequenceLength> [options]
    predict_stock.py --help
    predict_stock.py --listcompanies -k <keywords>
 
Options:
    -h --help                   help with predict_stock useage
    --plot= True/False          plot flag if true saves the plots in plot folder
    --saveModel= model-name     save_model flag if given saves the trained model with given model-name for future use
    -k <keywords>               search companies names using keyword
    --epochs = no-of-epochs     no of epochs

"""
import pandas as pd
from docopt import docopt
from datasets.raw_data_extractor import get_companies, get_intraday_dataset
from tsprediction.dataloader import dataloader_from_pandas
from tsprediction.train_model import train_model
from tsprediction.predict_model import predict_model
from figures.plot_data import plot_utils

if __name__ == "__main__":
    arguments = docopt(__doc__, version="0.0.rc1")
    if arguments["<predictSequenceLength>"]:
        sequence_length = int(arguments["<predictSequenceLength>"])

    if arguments["--listcompanies"] == True:
        companies_list = get_companies(arguments["-k"])["bestMatches"]
        assert len(companies_list) > 0, " cannot find any companies with that keyword"
        print("Symbol\t\tName")
        for companies in companies_list:
            print(f"{companies['1. symbol']}\t\t{companies['2. name']}")
        # test: check bestmatch key in json

    elif arguments["alphavantage"] == True:
        if arguments["--epochs"]:
            epochs = int(arguments["--epochs"])
        companySymbol = arguments["<companySymbol>"]
        predictSequenceLength = sequence_length
        plot_flag = arguments["--plot"]
        save_model = arguments["--saveModel"]
        dataset = get_intraday_dataset(companySymbol)
        
        dataset_train = dataset[: -2 * sequence_length]
        dataset_test = dataset[-2 * sequence_length :]

        train_test_data_tuple = dataloader_from_pandas(
            dataset_train,
            train=True,
            sequence_length=sequence_length,
            train_size_percentage=1
        )
        train_data_tuple = (train_test_data_tuple[0], train_test_data_tuple[1])
        if arguments["--epochs"]:
            model, loss = train_model(train_data_tuple, sequence_length, save_model=save_model, epochs = epochs)
        else:
            model, loss = train_model(
                train_data_tuple, sequence_length, save_model=save_model)


        print(f"loss of trained model = {loss}")

        # test_data_tuple=(train_test_data_tuple[2],train_test_data_tuple[3])
        train_test_data_tuple = dataloader_from_pandas(
            dataset_test,
            train=True,
            sequence_length=sequence_length,
            train_size_percentage=0
        )
        test_data_tuple = (train_test_data_tuple[2], train_test_data_tuple[3])

        predicted_data = predict_model(model, test_data_tuple[0])
        # print(test_data_tuple[0])
        plot_utils(test_data_tuple[1], predicted_data,sequence_length,dataset)
        

    elif arguments["custom"] == True:
        if arguments["--epochs"]:
            epochs = int(arguments["--epochs"])
        csvloc = arguments["<csvFileLocation>"]
        predictSequenceLength = sequence_length
        plot_flag = arguments["--plot"]
        save_model = arguments["--saveModel"]
        dataset = pd.read_csv(csvloc)

        dataset_train = dataset[: -2 * sequence_length]
        dataset_test = dataset[-2 * sequence_length :]

        train_test_data_tuple = dataloader_from_pandas(
            dataset_train,
            train=True,
            sequence_length=sequence_length,
            train_size_percentage=1,
            custom=True,
        )
        train_data_tuple = (train_test_data_tuple[0], train_test_data_tuple[1])
        if arguments["--epochs"]:
            model, loss = train_model(train_data_tuple, sequence_length, save_model=save_model, epochs = epochs)
        else:
            model, loss = train_model(
                train_data_tuple, sequence_length, save_model=save_model)


        print(f"loss of trained model = {loss}")

        # test_data_tuple=(train_test_data_tuple[2],train_test_data_tuple[3])
        train_test_data_tuple = dataloader_from_pandas(
            dataset_test,
            train=True,
            sequence_length=sequence_length,
            train_size_percentage=0,
            custom=True,
        )
        test_data_tuple = (train_test_data_tuple[2], train_test_data_tuple[3])

        predicted_data = predict_model(model, test_data_tuple[0])
        # print(test_data_tuple[0])
        plot_utils(test_data_tuple[1], predicted_data,sequence_length,dataset)
