import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from dataset import MonomerDataset
from model import Net, LinearRegression
import copy

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

def val_loop(model, dataloader, loss_fn):
    total_loss = 0.0
    num_batches = len(dataloader)

    for (X,y) in dataloader:
        pred = model(X)
        loss = loss_fn(pred,y)

        loss = loss.item()
        total_loss = total_loss + loss
    
    return total_loss/num_batches

def test_loop(model, dataloader, loss_fn, scaler, file_name):
    total_loss = 0.0
    num_batches = len(dataloader)

    pred_array = torch.zeros(num_batches)
    true_array = torch.zeros(num_batches)

    if file_name:
        out = open(file_name,"w")
        out.write("#predicted true\n")

    for i, (X,y) in enumerate(dataloader):
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

            pred_array[i] = pred_real
            true_array[i] = true_real
        
            out.write(str(true_real) + " " + str(pred_real) + "\n")
        
        loss = loss_fn(torch.tensor(pred_real),torch.tensor(true_real))

        loss = loss.item()
        total_loss = total_loss + loss
    
    if file_name:
        out.close()

    R2 = r2_score(true_array,pred_array)
    
    return math.sqrt(total_loss/num_batches), R2

N_epochs = 10000
patience = 500

data = pd.read_csv("surrogate_data.csv")

data = data.map(math.log)

train, test = train_test_split(data, test_size=0.2)
val, test = train_test_split(test, test_size=0.5)

scaler = MinMaxScaler()

scaler.fit(train)

train_scaled = scaler.transform(train)
val_scaled = scaler.transform(val)
test_scaled = scaler.transform(test)

train_dataset = MonomerDataset(train_scaled)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=100)

val_dataset = MonomerDataset(val_scaled)
val_dataloader = DataLoader(dataset=val_dataset,batch_size=100)

test_dataset = MonomerDataset(test_scaled)
test_dataloader = DataLoader(dataset=test_dataset,batch_size=1)

model = Net()
#model = LinearRegression()

loss_fn = nn.MSELoss()

optimizer = optim.Adam(model.parameters(),lr=1e-4)

best_val_loss = float('inf')
best_model_weights = None

loss_out = open("loss_curve.dat","w")
loss_out.write("#epoch train_loss val_loss\n")

for n in range(N_epochs):
    print(f"Epoch {n+1}")
    print("---------")
    train_loss = train_loop(model, train_dataloader, loss_fn, optimizer)
    val_loss = val_loop(model, val_dataloader, loss_fn)
    print(f"Training loss: {train_loss}")
    print(f"Validation loss: {val_loss}")

    loss_out.write(str(n+1) + " " + str(train_loss) + " " + str(val_loss) + "\n")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_weights = copy.deepcopy(model.state_dict())
        patience_curr = patience
    else:
        patience_curr -= 1
        if patience_curr == 0:
            break

loss_out.close()

model.load_state_dict(best_model_weights)

#train_loss, train_R2 = test_loop(model, train_dataloader, loss_fn, scaler, "train.dat")
test_loss, test_R2 = test_loop(model, test_dataloader, loss_fn, scaler, "test.dat")

#print(f"Training loss: {train_loss}")
#print(f"Training R^2: {train_R2}")

print(f"Test loss: {test_loss}")
print(f"Test R^2: {test_R2}")

with open("test_losses.dat","w") as f:
    f.write("Test loss: " + str(test_loss) + "\n")
    f.write("Test R^2: " + str(test_R2))

torch.save(model.state_dict(),"weights.pt")