"""
plots data
"""
import matplotlib.pyplot as plt
import joblib

      
def plot_utils(test_data,predicted_data,dataset=None):
    '''
    

    Parameters
    ----------
    test_data : TYPE
        DESCRIPTION.
    predicted_data : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''


    sc=joblib.load('models/norm_model.mod')
    data_predict = sc.inverse_transform(predicted_data)
    
    testY=test_data
    dataY = (testY[:50]).view(-1,1)
    dataY_plot = dataY.data.numpy()
    data_predict = sc.inverse_transform(predicted_data)
    dataY_plot = sc.inverse_transform(dataY_plot)
    plt.plot(dataY_plot)
    plt.plot(data_predict)
    plt.suptitle('Time-Series Prediction')
    plt.show()


