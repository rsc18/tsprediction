"""
predict model 
used for predicting and testing model
"""


def predict_model(model,test_data):
    '''
    
    Parameters
    ----------
    model : torch state_dict
        saved torch model from train_model. 
    test_data: test data x sequence

    Returns
    -------
    data_predict in normalized form

    '''
    testX=test_data
    model.eval()
    test_predict = model(testX[:50])
    tp = test_predict.view(-1, 1)
    data_predict = tp.data.numpy()
        
    return data_predict