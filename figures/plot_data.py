"""
plots data
"""
import time
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import numpy as np


category_dict = {
    "open": "1. open",
    "high": "2. high",
    "low": "3. low",
    "close": "4. close",
    "volume": "5. volume",
}


def plot_utils(
    test_data,
    predicted_data,
    seq,
    dataset=None,
    category="close",
    comapny_symbol="",
    save_plot=None,
    save_csv=None,
):
    """


    Parameters
    ----------
    test_data : array
         test data for prediction and comparison
    predicted_data : array
        predicted sequence

    Returns
    -------
    None.

    """
    date = "_".join(str(time.ctime()).split())
    filename = comapny_symbol + "_" + str(time.ctime())
    category = category if category else "close"
    sc2 = joblib.load("models/train_norm.mod")
    data_predict = sc2.inverse_transform(predicted_data)
    test_x = sc2.inverse_transform(test_data[0].view(-1, 1).data.numpy())
    test_y = test_data[1]
    data_y = (test_y).view(-1, 1)
    data_yplot = data_y.data.numpy()
    data_predict = sc2.inverse_transform(predicted_data)
    data_yplot = sc2.inverse_transform(data_yplot)
    data_predict = np.insert(data_predict, 0, test_x[-1], axis=0)

    plt.grid(True)
    plt.autoscale(axis="x", tight=True)

    if len(dataset.columns) > 2:
        dataset = dataset[category_dict[category]]

    if save_csv:
        save_data = pd.DataFrame()
        save_data["Input:"] = test_x.flatten()
        save_data["Real Output:"] = data_yplot.flatten()
        save_data["Predicted Output:"] = data_predict[1:, :].flatten()
        save_data.to_csv(
            "figures/prediction_{}.csv".format(filename), index=False, header=True
        )
    pp_x = list(dataset.index)[-seq - 1 :]
    line_index = list(dataset.index)[-seq - 1 :][0]
    plt.plot(dataset[-seq:], label="real data", color="navy")
    plt.plot(
        list(dataset.index)[-2 * seq : (-seq)],
        test_x,
        color="forestgreen",
        label="input_sequence",
    )
    plt.axvline(x=line_index, color="r")

    plt.plot(pp_x, data_predict.flatten(), label="output_sequence", color="peru")

    # plt.plot(data_yplot,label='real data')
    # plt.plot(data_predict, label='predict data')
    plt.xlabel("time")
    plt.ylabel(category)
    plt.title("Time-Series Prediction")
    plt.legend()
    if save_plot:
        plt.savefig("figures/{}.png".format(filename))
    plt.show()
