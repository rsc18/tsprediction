import sys
import os
import pandas as pd

CURRENT_DIR = os.getcwd()
PARENT_DIR = "/".join(CURRENT_DIR.split("/")[:-1])
sys.path.append(PARENT_DIR)

import datasets.raw_data_extractor as rde

dataset = rde.get_intraday_dataset(symbol, interval, key)
