"""AITW Monomer Surrogate Model definition"""
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn

# class OldNet(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.linear1 = nn.Linear(6,50)
#         self.relu1 = nn.ReLU()
#         self.linear2 = nn.Linear(50,50)
#         self.relu2 = nn.ReLU()
#         self.linear3 = nn.Linear(50,50)
#         self.relu3 = nn.ReLU()
#         self.linear4 = nn.Linear(50,1)

#         #self.relu4 = nn.ReLU()
#         self.sigmoid4 = nn.Sigmoid()

#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.relu1(x)
#         x = self.linear2(x)
#         x = self.relu2(x)
#         x = self.linear3(x)
#         x = self.relu3(x)
#         x = self.linear4(x)

#         #x = self.relu4(x)
#         x = self.sigmoid4(x)

#         return x

class Net(nn.Module):
    def __init__(self, scaler: MinMaxScaler = None):
        super().__init__()
        self.linear1 = nn.Linear(6,50)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(50,50)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(50,50)
        self.relu3 = nn.ReLU()
        self.linear4 = nn.Linear(50,1)

        #self.relu4 = nn.ReLU()
        self.sigmoid4 = nn.Sigmoid()

        scale = scaler.scale_ if scaler is not None else torch.ones(7)
        min_ = scaler.min_ if scaler is not None else torch.zeros(7)
        self.register_buffer("scaler_scale", torch.tensor(scale, dtype=torch.float32))
        self.register_buffer("scaler_min", torch.tensor(min_, dtype=torch.float32))

    def forward(self, x):
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
