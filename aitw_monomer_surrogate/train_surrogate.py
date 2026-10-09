"""AITW Monomer Surrogate training script"""
import copy
import math
import os

import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader

from .dataset import MonomerDataset
from .model import LinearRegression, Net


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

def train_surrogate(
        n_epochs: int = 10000,
        patience: int = 500,
        data_file: str = "surrogate_data.csv",
        loss_curve_file: str = "loss_curve.dat",
        test_losses_file: str = "test_losses.dat",
        test_file: str = "test.dat",
        output_weights_file: str = "weights.pt",
        train_test_split_ratio: float = 0.2,
        val_test_split_ratio: float = 0.5,
        train_batch_size: int = 100,
        val_batch_size: int = 100,
        test_batch_size: int = 1,
    ):
    """Runs the training of the surrogate model.

    Args:
        n_epochs (int, optional): The number of epochs to train the model. Defaults to 10000.
        patience (int, optional): The number of epochs to wait before stopping the training if no improvement is observed. Defaults to 500.
        data_file (str, optional): The path to the CSV file containing the training data. Defaults to "surrogate_data.csv".
        loss_curve_file (str, optional): The path to the file where the loss curve will be saved. Defaults to "loss_curve.dat".
        test_losses_file (str, optional): The path to the file where the test losses will be saved. Defaults to "test_losses.dat".
        test_file (str, optional): The path to the file where the test results will be saved. Defaults to "test.dat".
        output_weights_file (str, optional): The path to the file where the trained weights will be saved. Defaults to "weights.pt".
        train_test_split_ratio (float, optional): The ratio of the training data to the test data. Defaults to 0.2.
        val_test_split_ratio (float, optional): The ratio of the validation data to the test data. Defaults to 0.5.
        train_batch_size (int, optional): The batch size for training. Defaults to 100.
        val_batch_size (int, optional): The batch size for validation. Defaults to 100.
        test_batch_size (int, optional): The batch size for testing. Defaults to 1.
    """

    data = pd.read_csv(data_file)

    data = data.map(math.log)

    train, test = train_test_split(data, test_size=train_test_split_ratio)
    val, test = train_test_split(test, test_size=val_test_split_ratio)

    scaler = MinMaxScaler()

    scaler.fit(train)

    train_scaled = scaler.transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)

    train_dataset = MonomerDataset(train_scaled)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size)

    val_dataset = MonomerDataset(val_scaled)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size)

    test_dataset = MonomerDataset(test_scaled)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size)

    model = Net()
    #model = LinearRegression()

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),lr=1e-4)

    best_val_loss = float('inf')
    best_model_weights = None

    basedir = os.path.dirname(loss_curve_file)
    os.makedirs(basedir, exist_ok=True)
    with open(loss_curve_file, "w", encoding="utf-8") as loss_out:
        loss_out.write("#epoch train_loss val_loss\n")

        for n in range(n_epochs):
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

    model.load_state_dict(best_model_weights)

    #train_loss, train_R2 = test_loop(model, train_dataloader, loss_fn, scaler, "train.dat")
    basedir = os.path.dirname(test_file)
    os.makedirs(basedir, exist_ok=True)
    test_loss, test_R2 = test_loop(model, test_dataloader, loss_fn, scaler, test_file)

    #print(f"Training loss: {train_loss}")
    #print(f"Training R^2: {train_R2}")

    print(f"Test loss: {test_loss}")
    print(f"Test R^2: {test_R2}")

    basedir = os.path.dirname(test_losses_file)
    os.makedirs(basedir, exist_ok=True)
    with open(test_losses_file, "w", encoding="utf-8") as f:
        f.write("Test loss: " + str(test_loss) + "\n")
        f.write("Test R^2: " + str(test_R2))

    basedir = os.path.dirname(output_weights_file)
    os.makedirs(basedir, exist_ok=True)
    torch.save(model.state_dict(), output_weights_file)
