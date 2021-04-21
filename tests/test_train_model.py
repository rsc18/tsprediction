"""
Test module for train model
"""
import pandas as pd
import tsprediction.train_model as tm
import tsprediction.dataloader as dl


def test_train_model():
    """
    Returns
    -------
    None.

    """
    dataset_train = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    train_test_data_tuple = dl.dataloader_from_pandas(
        dataset_train, sequence_length=2, train_size_percentage=1, custom=True
    )
    train_data_tuple = (train_test_data_tuple[0], train_test_data_tuple[1])

    _, loss = tm.train_model(train_data_tuple, 2, epochs=2)
    assert loss > 0
    assert isinstance(loss, float)
