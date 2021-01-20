# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 16:00:29 2021

@author: rsc18
"""


from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from alpha_vantage.sectorperformance import SectorPerformances
from alpha_vantage.cryptocurrencies import CryptoCurrencies
import matplotlib
import matplotlib.pyplot as plt
import os

matplotlib.rcParams['figure.figsize'] = (20.0, 10.0)


ts = TimeSeries(key='2JVYQIAGV1ABJD0O', output_format='pandas')
#Microsoft : MSFT
#Apple : AAPL 
data_apple, meta_data_ms = ts.get_intraday(symbol='AAPL',interval='60min', outputsize='full')
# We can describe it
# data.describe()
data_ms, meta_data_ms = ts.get_intraday(symbol='MSFT',interval='60min', outputsize='full')
#LI
data_li, meta_data_li = ts.get_intraday(symbol='LI',interval='60min', outputsize='full')

fig,axs = plt.subplots(3)
data_apple['4. close'].plot(ax=axs[0])
data_ms['4. close'].plot(ax=axs[1])
data_li['4. close'].plot(ax=axs[2])

