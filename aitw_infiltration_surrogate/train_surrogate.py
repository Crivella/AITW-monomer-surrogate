"""AITW Monomer Infiltration Surrogate training script"""
import copy
import logging
import math
import os
import sys

import pandas as pd
import torch
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader

from .dataset import MonomerDataset
from .loggers import get_logger, set_console_level
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

def test_loop(model, dataloader, loss_fn, file_name):
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
            true_real = y.detach().item()
            true_real = math.exp(true_real)
            pred_real = torch.zeros(7).reshape(1,-1)

            pred = pred.detach()
            pred_real[0,6] = pred
            pred_real = math.exp(pred_real[0,6])

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
        output_dir: str = "output",

        train_test_split_ratio: float = 0.2,
        val_test_split_ratio: float = 0.5,
        train_batch_size: int = 100,
        val_batch_size: int = 100,
        test_batch_size: int = 1,

        loss_threshold: float = None,
        save_interval: int = 0,

        log_level: int = logging.INFO,
    ):
    """Runs the training of the surrogate model.

    Args:
        n_epochs (int, optional): The number of epochs to train the model. Defaults to 10000.
        patience (int, optional): The number of epochs to wait before stopping the training if no improvement is observed. Defaults to 500.
        data_file (str, optional): The path to the CSV file containing the training data. Defaults to "surrogate_data.csv".
        output_dir (str, optional): The directory where the output files will be saved. Defaults to "output".
    
        train_test_split_ratio (float, optional): The ratio of the training data to the test data. Defaults to 0.2.
        val_test_split_ratio (float, optional): The ratio of the validation data to the test data. Defaults to 0.5.
        train_batch_size (int, optional): The batch size for training. Defaults to 100.
        val_batch_size (int, optional): The batch size for validation. Defaults to 100.
        test_batch_size (int, optional): The batch size for testing. Defaults to 1.

        loss_threshold (float, optional): The threshold for the loss function. Defaults to None (no threshold).
        save_interval (int, optional): The interval at which to save the model weights. Defaults to 0 (no saving).

        log_level (str, optional): The logging level. Defaults to "INFO".
    """
    if os.path.exists(output_dir):
        print(f"Output directory '{output_dir}' already exists. Please choose a different directory or remove the existing one.")
        sys.exit(1)
    os.makedirs(output_dir, exist_ok=True)

    logfile = os.path.join(output_dir, "training.log")
    loss_curve_file = os.path.join(output_dir, "loss_curve.dat")
    test_losses_file = os.path.join(output_dir, "test_losses.dat")
    test_file = os.path.join(output_dir, "test.dat")
    output_weights_file = os.path.join(output_dir, "weights.pt")

    logger = get_logger("train_surrogate", logfile)
    set_console_level(logger, log_level)

    logger.info(f"Training surrogate model with the following parameters:")
    logger.info(f"Number of epochs: {n_epochs}")
    logger.info(f"Patience: {patience}")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training/test split ratio: {train_test_split_ratio}")
    logger.info(f"Validation/test split ratio: {val_test_split_ratio}")
    logger.info(f"Training batch size: {train_batch_size}")
    logger.info(f"Validation batch size: {val_batch_size}")
    logger.info(f"Testing batch size: {test_batch_size}")
    logger.info(f"Loss threshold: {loss_threshold}")
    logger.info(f"Save interval: {save_interval}")

    data = pd.read_csv(data_file)

    data = data.map(math.log)

    train, test = train_test_split(data, test_size=train_test_split_ratio)
    val, test = train_test_split(test, test_size=val_test_split_ratio)

    scaler = MinMaxScaler()

    scaler.fit(train)

    train_dataset = MonomerDataset(train.to_numpy())
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size)

    val_dataset = MonomerDataset(val.to_numpy())
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=val_batch_size)

    test_dataset = MonomerDataset(test.to_numpy())
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size)

    model = Net(scaler)
    #model = LinearRegression()

    loss_fn = nn.MSELoss()

    optimizer = optim.Adam(model.parameters(),lr=1e-4)

    best_val_loss = float('inf')
    best_model_weights = None

    with open(loss_curve_file, "w", encoding="utf-8") as loss_out:
        loss_out.write("#epoch train_loss val_loss\n")

        patience_curr = patience
        for n in range(n_epochs):
            if n == 0 or (n+1) % 100 == 0:
                log_fn = logger.info
            else:
                log_fn = logger.debug
            log_fn(f"Epoch {n+1} " + "-" * 20)
            train_loss = train_loop(model, train_dataloader, loss_fn, optimizer)
            val_loss = val_loop(model, val_dataloader, loss_fn)
            log_fn(f"Patience remaining: {patience_curr}")
            log_fn(f"Training loss: {train_loss}")
            log_fn(f"Validation loss: {val_loss}")

            loss_out.write(str(n+1) + " " + str(train_loss) + " " + str(val_loss) + "\n")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())
                patience_curr = patience
                if loss_threshold is not None and val_loss < loss_threshold:
                    logger.info(f"Early stopping triggered by loss threshold after {n+1} epochs.")
                    break
            else:
                patience_curr -= 1
                if patience_curr == 0:
                    logger.info(f"Early stopping triggered by patience after {n+1} epochs.")
                    break

            if save_interval > 0 and (n+1) % save_interval == 0:
                file_path = os.path.join(output_dir, f"model_epoch_{n+1}.pt")
                logger.debug(f"Saving model weights to {file_path}")
                torch.save(model.state_dict(), file_path)

    model.load_state_dict(best_model_weights)

    # train_loss, train_R2 = test_loop(model, train_dataloader, loss_fn, scaler, "train.dat")
    test_loss, test_R2 = test_loop(model, test_dataloader, loss_fn, test_file)

    logger.info(f"Test loss: {test_loss}")
    logger.info(f"Test R^2: {test_R2}")

    with open(test_losses_file, "w", encoding="utf-8") as f:
        f.write("Test loss: " + str(test_loss) + "\n")
        f.write("Test R^2: " + str(test_R2))

    torch.save(model.state_dict(), output_weights_file)

    logger.info(f"Trained weights saved to {output_weights_file}")
