"""
Test module for predict model
"""
import pandas as pd
import tsprediction.train_model as tm
import tsprediction.predict_model as pm
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

    model, _ = tm.train_model(train_data_tuple, 2, epochs=2)

    dataset_test = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    train_test_data_tuple = dl.dataloader_from_pandas(
        dataset_test, sequence_length=2, train_size_percentage=0, custom=True
    )
    test_data = train_test_data_tuple[2]
    predicted_data = pm.predict_model(model, test_data)
    assert len(predicted_data) == 14
