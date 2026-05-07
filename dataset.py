import pandas as pd
import torch
from torch.utils.data import Dataset

class MonomerDataset(Dataset):
    def __init__(self,data):
        self.X = data[:,0:6]
        self.y = data[:,6]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return torch.tensor(self.X[index],dtype=torch.float32), torch.tensor(self.y[index],dtype=torch.float32).unsqueeze(-1)