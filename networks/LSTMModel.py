import torch.nn as nn
import torch
from argparse import ArgumentParser
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
from torch.utils.data import TensorDataset, DataLoader
import logging
import time
import numpy as np
import gc


class Module_LSTM(nn.Module, BaseEstimator):
    def __init__(self, hidden_size=16, dropout=0.1):
        super().__init__()
        self.input_size = 38
        self.hidden_size = hidden_size
        self.num_layers = 1
        self.output_size = 7
        self.batch_size = 128
        self.dropout = dropout
        self.lstm = nn.LSTM(self.input_size + 7, self.hidden_size,
                            self.num_layers, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(self.hidden_size, 16)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(16, self.output_size)

    def forward(self, x, y_prev):
        x1 = torch.cat((x, y_prev), dim=2)
        output, _ = self.lstm(x1)  # output(5, 30, 64)
        output = self.dropout1(output)
        output = output[:, -1, :]
        bs = int(len(output))
        output = self.linear1(output.reshape(bs, -1))  # (5, 30, 1)
        output = self.dropout2(output)
        pred = self.linear2(output)
        return pred
