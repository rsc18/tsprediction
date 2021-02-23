'''
Train model here
'''
import torch 
from tsprediction.lstm_class import LSTM

def train_model(train_dataset_tuple,sequence_length,save_model=None, epochs=1000):
    '''
    

    Parameters
    ----------
    train_dataset_tuple : train data tuple 
        DESCRIPTION.
    sequence_length : TYPE
        DESCRIPTION.
    save_model : TYPE
        DESCRIPTION.

    Returns
    -------
    lstm : TYPE
        DESCRIPTION.

    '''
    debug=True
    
    trainX,trainY=train_dataset_tuple
    num_epochs = epochs
    learning_rate = 0.01
    
    input_size = 1
    hidden_size = 8
    num_layers = 1
    
    num_classes = sequence_length
    
    lstm = LSTM(num_classes, input_size, hidden_size, num_layers)
    
    criterion = torch.nn.MSELoss()    # mean-squared error for regression
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        outputs = lstm(trainX)
        optimizer.zero_grad()
        # obtain the loss function
        loss = criterion(outputs, trainY.view(trainY.shape[0], -1))
        
        loss.backward()
        
        optimizer.step()
        if debug:
            if epoch % 20 == 0:
                # print(outputs)
                # print(trainY.view(trainY.shape[0], -1))
                print("Epoch: %d, loss: %1.5f" % (epoch, loss.item()))
    if save_model:        
        torch.save(lstm.state_dict(),'models/'+save_model)
    return lstm,loss.item()