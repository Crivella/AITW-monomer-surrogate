import torch
from torch import nn

class Net(nn.Module):
    def __init__(self):
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

    def forward(self,x):
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.linear4(x)

        #x = self.relu4(x)
        x = self.sigmoid4(x)

        return x
    
class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(6,1)

    def forward(self,x):
        x = self.linear(x)

        return x
