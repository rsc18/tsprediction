# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 15:31:02 2021
Sample pytorch lstm program
@author: rsc18
"""

import glob
import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch

df=pd.read_csv("trade_20141122.csv")
df = df[df.symbol == 'XBTUSD']
df.timestamp = pd.to_datetime(df.timestamp.str.replace('D', 'T')) # covert to timestamp type
df = df.sort_values('timestamp')
df.set_index('timestamp', inplace=True) # set index to timestamp
df.head()