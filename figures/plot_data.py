"""
plots data
"""
import matplotlib.pyplot as plt
import joblib

      
def plot_utils(test_data,predicted_data,seq,dataset=None):
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
    
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    
    if len(dataset.columns)>2:
        dataset=dataset['4. close']
        x = list(dataset.index)[0:seq]
        plt.plot(dataset, label='real data')
        plt.axvline(x = x[-1], color = 'r') 
    else:
        x = list(dataset.index)[-seq:]
        plt.plot(dataset, label='real data')
        plt.axvline(x = x[0], color = 'r')         
    
    plt.plot(x,data_predict.flatten(), label='predict data')
    
    # plt.plot(dataY_plot,label='real data')
    # plt.plot(data_predict, label='predict data')
    plt.xlabel('Time')
    plt.ylabel('Close Value')
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()

