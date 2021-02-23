"""
Normalize time series data
"""

from sklearn.preprocessing import MinMaxScaler
import joblib

def norm_data(dataset,norm_name):
    '''
    

    Parameters
    ----------
    dataset : pandas series
        DESCRIPTION.

    Returns
    -------
    normalized_data : TYPE
        DESCRIPTION.

    '''
    min_max_scalar = MinMaxScaler()
    normalized_data = min_max_scalar.fit_transform(dataset)
    joblib.dump(min_max_scalar,f'models/{norm_name}.mod')
    return normalized_data