"""
plots data
"""
import matplotlib.pyplot as plt
import joblib
import numpy as np

category_dict = {'open' : '1. open',
                  'high' : '2. high',
                  'low' : '3. low',
                  'close' : '4. close',
                  'volume' : '5. volume'}
      
def plot_utils(test_data,predicted_data,seq,dataset=None, category='close'):
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
    category = category if category else 'close'
    sc2=joblib.load('models/test_norm.mod')
    data_predict = sc2.inverse_transform(predicted_data)    
    testX=sc2.inverse_transform(test_data[0].view(-1,1).data.numpy())
    testY=test_data[1]
    dataY = (testY).view(-1,1)    
    dataY_plot = dataY.data.numpy()
    data_predict = sc2.inverse_transform(predicted_data)
    dataY_plot = sc2.inverse_transform(dataY_plot)
    data_predict = np.insert(data_predict, 0, testX[-1], axis=0)
    
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    
    if len(dataset.columns)>2:
        dataset=dataset[category_dict[category]]

    x = list(dataset.index)[-seq-1:]
    line_index=list(dataset.index)[-seq-1:][0]
    plt.plot(dataset, label='real data',color='navy')
    plt.plot(list(dataset.index)[-2*seq:(-seq)],testX,color='forestgreen',label='input_sequence')
    plt.axvline(x = line_index, color = 'r')         
    
    plt.plot(x,data_predict.flatten(), label='output_sequence',color='peru')
    
    # plt.plot(dataY_plot,label='real data')
    # plt.plot(data_predict, label='predict data')
    plt.xlabel('time')
    plt.ylabel(category)
    plt.title('Time-Series Prediction')
    plt.legend()
    plt.show()
