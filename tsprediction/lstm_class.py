"""
Build LSTM model
pylint bug: https://github.com/microsoft/vscode-python/issues/5635
"""

import torch
import torch.nn as nn
from torch.autograd import Variable


class LSTM(nn.Module):
    """
    This is a LSTM class
    """

    def __init__(self, num_classes, input_size, hidden_size, num_layers):
        """
        Parameters
        ----------
        num_classes : TYPE
            DESCRIPTION.
        input_size : TYPE
            DESCRIPTION.
        hidden_size : TYPE
            DESCRIPTION.
        num_layers : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.seq_length = seq_length

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # print()
        self.f_c = nn.Linear(hidden_size, num_classes)

        # self.relu = nn.ReLU()
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, data):
        """
        Parameters
        ----------
        x : TYPE
            DESCRIPTION.

        Returns
        -------
        out : TYPE
            DESCRIPTION.

        """
        dev = torch.device("cpu")
        if torch.cuda.is_available():
            dev = torch.device("cuda")

        h_0 = Variable(torch.zeros(self.num_layers, data.size(0), self.hidden_size)).to(
            dev
        )

        c_0 = Variable(torch.zeros(self.num_layers, data.size(0), self.hidden_size)).to(
            dev
        )
        # Propagate input through LSTM
        _, (h_out, _) = self.lstm(data, (h_0, c_0))

        # print(h_out.shape)
        h_out = h_out[-1].view(-1, self.hidden_size)
        # print(h_out.shape)
        out = self.f_c(h_out)
        # out = self.relu(out)
        out = self.leaky_relu(out)

        return out
