import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import MonomerDataset
from model import Net, LinearRegression

def train_loop(model, dataloader, loss_fn, optimizer):
    total_loss = 0.0
    num_batches = len(dataloader)

    for (X,y) in dataloader:
        pred = model(X)
        loss = loss_fn(pred,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        loss = loss.item()
        total_loss = total_loss + loss
    
    return total_loss/num_batches

def val_loop(model, dataloader, loss_fn, scaler, file_name):
    total_loss = 0.0
    num_batches = len(dataloader)

    if file_name:
        out = open(file_name,"w")

    for (X,y) in dataloader:
        pred = model(X)

        if file_name:
            true_real = scaler.inverse_transform(torch.cat((X,y),1))[0,6]
            true_real = math.exp(true_real)
            pred_real = torch.zeros(7).reshape(1,-1)

            pred = pred.detach()
            pred_real[0,6] = pred
            pred_real = scaler.inverse_transform(pred_real)
            pred_real = math.exp(pred_real[0,6])
            #pred_real = pred_real[0,6]
        
            out.write(str(true_real) + " " + str(pred_real) + "\n")

        loss = loss_fn(torch.tensor(pred_real),torch.tensor(true_real))

        loss = loss.item()
        total_loss = total_loss + loss
    
    if file_name:
        out.close()
    
    return math.sqrt(total_loss/num_batches)

N_epochs = 10000

data = pd.read_csv("surrogate_data.csv")

data = data.map(math.log)

#data['k_long'] = data['k_long'].apply(math.log)
#data['mu_value'] = data['mu_value'].apply(math.log)
#data['sat_time'] = data['sat_time'].apply(math.log)

train, test = train_test_split(data, test_size=0.2)

scaler = MinMaxScaler()

scaler.fit(train)

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

train_dataset = MonomerDataset(train_scaled)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=1)

test_dataset = MonomerDataset(test_scaled)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)

model = Net()
#model = LinearRegression()

loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(),lr=1e-4)

for n in range(N_epochs):
    print(f"Epoch {n+1}")
    print("---------")
    train_loss = train_loop(model, train_dataloader, loss_fn, optimizer)
    print(train_loss)

train_dataloader = DataLoader(dataset=train_dataset,batch_size=1)

train_loss = val_loop(model, train_dataloader, loss_fn, scaler, "train.dat")
test_loss = val_loop(model, test_dataloader, loss_fn, scaler, "test.dat")

print(train_loss)
print(test_loss)