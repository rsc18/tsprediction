# -*- coding: utf-8 -*-
"""
tsprediction
--------------------

This module extracts data of company with interval 60 min default given company symbol.
"""
from alpha_vantage.timeseries import TimeSeries


def get_intraday_dataset(symbol, interval="60min", key="2JVYQIAGV1ABJD0O"):
    """
    Get intraday stock values for a given symbol

    :Args:
        symbol: the company keyword for which we want the stock data
    :Kwargs:
        interval: 1min, 5min, 15min, 30min, 60min 
                    (the interval between the data points)\
        key: API key from the user
    :Returns:
        a dataframe of stock datas with a given interval
    :Raises:
    """
    ts = TimeSeries(key=key, output_format="pandas")
    dataset, meta_data = ts.get_intraday(
        symbol=symbol, interval=interval, outputsize="full"
    )
    return dataset


# =============================================================================
#
# def get_companies(keywords, function='SYMBOL_SEARCH', key='2JVYQIAGV1ABJD0O'):
#     API_URL = "https://www.alphavantage.co/query"
#
#     data = {
#         "function": function,
#         "keywords": keywords,
#         "apikey": key,
#         }
#
#     response = requests.get(API_URL, params=data)
#
#     return response.json()
# =============================================================================


# https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=tesco&apikey=demo
