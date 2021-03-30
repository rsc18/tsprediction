#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:31:47 2021

@author: chashi
"""
import sys
import os
import pandas as pd

CURRENT_DIR = os.getcwd()
PARENT_DIR = "/".join(CURRENT_DIR.split("/")[:-1])
sys.path.append(PARENT_DIR)

import datasets.raw_data_extractor as rde 

def test_get_intraday_dataset():
    """Tests get_intraday_dataset function
    Args:

    Kwargs:

    Returns:

    Raises:
    """
    symbol = "LI"
    key = "2JVYQIAGV1ABJD0O"
    interval = "60min"
    dataset = rde.get_intraday_dataset(symbol, interval, key)
    assert len(dataset) > 0
    assert dataset.columns[3] == "4. close"
    assert pd.core.dtypes.common.is_datetime_or_timedelta_dtype(dataset.index)
