"""
Normalize time series data
"""

from sklearn.preprocessing import MinMaxScaler

# from sklearn.preprocessing import StandardScaler
import joblib


def norm_data(dataset, norm_name):
    """
    Parameters
    ----------
    dataset : pandas series
        data to normalize

    Returns
    -------
    normalized_data : TYPE
        normalized data

    """
    if norm_name == "test_norm":
        sc2 = joblib.load("models/train_norm.mod")
        return sc2.transform(dataset)

    # std_scaler=StandardScaler()
    # normalized_data=std_scaler.fit_transform(dataset)
    # joblib.dump(std_scaler, f'models/{norm_name}.mod')

    min_max_scalar = MinMaxScaler()
    normalized_data = min_max_scalar.fit_transform(dataset)
    joblib.dump(min_max_scalar, f"models/{norm_name}.mod")
    return normalized_data
