"""
This module is used to predict the sequence given a model and input sequence
"""

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
    model.eval()
    test_predict = model(test_data[:50])
    t_p = test_predict.view(-1, 1)
    data_predict = t_p.data.numpy()
    return data_predict
