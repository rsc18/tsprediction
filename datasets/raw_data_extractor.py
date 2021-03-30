# -*- coding: utf-8 -*-
"""
This module extracts data of company with interval 60 min default given company symbol.
"""
from alpha_vantage.timeseries import TimeSeries
import requests


def get_intraday_dataset(symbol, interval="30min", key="2JVYQIAGV1ABJD0O"):
    """
    Gets intraday stock values for a given symbol
    Parameters
    ----------
    symbol : string
        The company keyword for which we want the stock data.
    interval : string, optional
        1min, 5min, 15min, 30min, 60min
                    (the interval between the data points). The default is "30min".
    key : string, optional
        API key from the user. The default is "2JVYQIAGV1ABJD0O".

    Returns
    -------
    returns: dataframe
        A dataframe of stock datas with a given interval

    """
    t_s = TimeSeries(key=key, output_format="pandas")
    dataset = t_s.get_intraday(symbol=symbol, interval=interval, outputsize="full")
    return dataset[0]


def get_companies(keywords, function="SYMBOL_SEARCH", key="2JVYQIAGV1ABJD0O"):
    """
    Gets the list of companies whose name matches with the keyword
    Parameters
    ----------
    keywords : string
        A keyword for company search
    function : string, optional
       The function of the API The default is 'SYMBOL_SEARCH'.
    key : string, optional
        User key for accessing the alpha vantage API. The default is '2JVYQIAGV1ABJD0O'.

    Returns
    -------
    Reurns: JSON
        JSON list of companies whose name has matched with the keyword

    """
    # test: check bestmatch key in json

    api_url = "https://www.alphavantage.co/query"

    data = {
        "function": function,
        "keywords": keywords,
        "apikey": key,
    }

    response = requests.get(api_url, params=data)

    return response.json()
