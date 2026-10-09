"""AITW Monomer Infiltration Surrogate Model definition"""
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn


class Net(nn.Module):
    def __init__(self, scaler: MinMaxScaler = None, logger=None):
        super().__init__()
        self.logger = logger
        self.linear1 = nn.Linear(6,50)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(50,50)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(50,50)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(50,1)

        #self.relu4 = nn.ReLU()
        self.sigmoid4 = nn.Sigmoid()

        if scaler is None:
            scale = torch.ones(7)
            min_ = torch.zeros(7)
            data_min = torch.zeros(7)
            data_max = torch.ones(7)
        else:
            scale = torch.tensor(scaler.scale_, dtype=torch.float32)
            min_ = torch.tensor(scaler.min_, dtype=torch.float32)
            data_min = torch.tensor(scaler.data_min_, dtype=torch.float32)
            data_max = torch.tensor(scaler.data_max_, dtype=torch.float32)
        self.register_buffer("scaler_scale", scale)
        self.register_buffer("scaler_min", min_)
        self.register_buffer("scaler_data_min", data_min)
        self.register_buffer("scaler_data_max", data_max)

    def forward(self, x):
        if torch.any(x < self.scaler_data_min[:6]) or torch.any(x > self.scaler_data_max[:6]):
            if self.logger:
                self.logger.warning("Input data is out of the range of the scaler used for training.")
            else:
                print("WARNING: Input data is out of the range of the scaler used for training.")
        x = x * self.scaler_scale[:6] + self.scaler_min[:6]  # Inverse transform the input data

        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)

        #x = self.relu4(x)
        x = self.sigmoid4(x)

        # Inverse transform the output data
        x = (x - self.scaler_min[6]) / self.scaler_scale[6]  # Inverse transform the output data

        return x


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6,1)

    def forward(self,x):
        x = self.linear(x)

        return x
