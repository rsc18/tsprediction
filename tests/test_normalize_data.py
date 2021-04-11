"""
Test module for normalize data
"""
import pandas as pd
import joblib
from pandas._testing import assert_frame_equal
from tsprediction.normalize_data import norm_data


def test_norm_data():
    '''
    1. Create normalization model and save it to models folder
    2. Load that normalization model and normalized data
    3. Testing data == unnormalized (normalizaed (data))

    Returns
    -------
    None.

    '''
    dataset = pd.DataFrame([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    n_data = norm_data(dataset, "train_norm")
    model = joblib.load("models/train_norm.mod")
    ops_dataset = model.inverse_transform(n_data)
    ops_dataset = pd.DataFrame(ops_dataset)
    assert_frame_equal(ops_dataset, dataset)
