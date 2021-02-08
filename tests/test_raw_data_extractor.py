#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 15:31:47 2021

@author: chashi
"""
import unittest
import raw_data_extractor as rde
import pandas as pd

class data_extractor_test_case(unittest.TestCase):
    
    def test_get_intraday_dataset(self):
        '''

        '''
        symbol= 'LI'
        key = '2JVYQIAGV1ABJD0O'
        interval = '60min'
        dataset = rde.get_intraday_dataset(symbol, interval, key)
        assert len(dataset) > 0
        assert dataset.columns[3] == '4. close'
        assert pd.core.dtypes.common.is_datetime_or_timedelta_dtype(dataset.index)
if __name__ == '__main__':
    unittest.main()