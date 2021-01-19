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
data, meta_data = ts.get_intraday(symbol='MSFT',interval='1min', outputsize='full')
# We can describe it
data.describe()


data['4. close'].plot()
plt.title('Intraday Times Series for the MSFT stock (1 min)')
plt.grid()
plt.show()
