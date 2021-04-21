"""
Test module for dataloader
"""
import pandas as pd
import tsprediction.dataloader as dl


def test_get_input_and_target():
    """
    Returns
    -------
    None.

    """
    dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data_all = dl.get_input_and_target(dataset, 2)

    print(data_all)
    assert len(data_all) == 7
    assert data_all[0][0][0] == 1
    assert data_all[0][1][0] == 3


def test_shuffled_input_target():
    """
    Returns
    -------
    None.

    """
    dataset = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    data_all = dl.get_input_and_target(dataset, 2)
    inp, out = dl.shuffled_input_target(data_all)
    assert len(inp[0]) == 2
    assert len(out[0]) == 2
    assert len(inp) == len(data_all)


def test_dataloader_from_pandas():
    """
    Returns
    -------
    None.

    """
    tr_df = pd.DataFrame([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    train_test_data_tuple = dl.dataloader_from_pandas(tr_df, 1, 2, custom=True)
    assert tr_df.size == 10
    assert len(train_test_data_tuple[0]) == 7
