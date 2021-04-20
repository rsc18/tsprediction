import sys
import os
import pandas as pd
CURRENT_DIR = os.getcwd()
PARENT_DIR = "/".join(CURRENT_DIR.split("/")[:-1])
sys.path.append(PARENT_DIR)
import tsprediction.train_model as tm
import tsprediction.dataloader as dl

def test_train_model():
    dataset_train = pd.DataFrame([1,2,3,4,5,6,7,8,9,10])
    train_test_data_tuple = dl.dataloader_from_pandas(
            dataset_train,
            sequence_length=2,
            train_size_percentage=1,
            custom=True,
        )
    train_data_tuple = (train_test_data_tuple[0], train_test_data_tuple[1])
    
    model, loss = tm.train_model(
                train_data_tuple, 2, epochs=2
            )
    assert len(loss) == len(train_data_tuple[0])
    assert type(model) == 'tsprediction.lstm_class.LSTM'

