"""
This module is used to predict the sequence given a model and input sequence
pylint bug: https://github.com/pytorch/pytorch/issues/701
"""
import torch

def predict_model(model, test_data):
    """

    Parameters
    ----------
    model : torch state_dict
        saved torch model from train_model.
    test_data: test data x sequence

    Returns
    -------
    data_predict in normalized form

    """
    dev=torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
    model=model.to(dev)
    test_data=test_data.to(dev)
    model.eval()
    test_predict = model(test_data[:50])
    t_p = test_predict.view(-1, 1)
    t_p=t_p.to('cpu')
    data_predict = t_p.data.numpy()
    return data_predict
