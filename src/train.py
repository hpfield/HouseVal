"""
Script to conduct regression training based on command line arguments
--model: model to train
--train_data: data to train on
--test_data: data to test on
--epochs: number of epochs to train for
--batch_size: batch size for training
--learning_rate: learning rate for training
--architecture: architecture of the model
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

import sklearn as sk
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
# from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from architectures.MLPRegressor import MLPRegressor



src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(src_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="linear", help="Model to train")
    parser.add_argument("--train_data", type=str, default="train_numerical_mean_imputed.csv", help="Data to train on")
    parser.add_argument("--test_data", type=str, default="test_numerical_mean_imputed.csv", help="Data to test on")
    parser.add_argument("--epochs", type=int, default=1000, help="Number of epochs to train for")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate for training")
    parser.add_argument("--architecture", type=str, default="mlp", help="Architecture of the model")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load the training data
    logger.info(f"Loading data from {args.train_data}")
    data = pd.read_csv("data/" + args.train_data)

    # Split the data into features and target
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    # Load the test data
    logger.info(f"Loading data from {args.test_data}")
    test_data = pd.read_csv("data/" + args.test_data)

    X = X.drop("Id", axis=1)
    X_test = test_data.drop("Id", axis=1)

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=100)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    # Convert the data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train.values, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val.values, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    input_size = X_train.shape[1]

    # Use RandomForestRegressor
    if args.model == "random_forest":
        model = RandomForestRegressor(n_estimators=100, random_state=100)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        logger.info(f"Root mean squared error: {rmse}")
    elif args.model == "mlp":
        model = MLPRegressor(input_size=input_size, hidden_layers=[100, 50], output_size=1)

        criterion = nn.MSELoss()

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        writer = SummaryWriter()

        for epoch in range(args.epochs):
            model.train()
            for X_batch, y_batch in train_loader:
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch.view(-1, 1))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            model.eval()
            if (epoch+1) % 100 == 0:
                with torch.no_grad():
                    loss = 0
                    for X_batch, y_batch in train_loader:
                        y_pred = model(X_batch)
                        loss += criterion(y_pred, y_batch.view(-1, 1))
                    loss /= len(train_loader)
                    writer.add_scalar("Loss/train", loss, epoch)
                    print(f"Epoch {epoch+1}, train: \t\t{loss}")
                    val_loss = 0
                    for X_val_batch, y_val_batch in val_loader:
                        y_val_pred = model(X_val_batch)
                        val_loss += criterion(y_val_pred, y_val_batch.view(-1, 1))
                    val_loss /= len(val_loader)
                    writer.add_scalar("Loss/val", val_loss, epoch)
                    print(f"Epoch {epoch+1}, val: \t\t{val_loss}")
                
                # Save checkpoint
                torch.save(model, f"models/{args.architecture}_{args.model}_{time.strftime('%Y_%m_%d_%H:%M:%S')}_epoch{epoch+1}.pt")

        writer.close()

    # Make predictions
    logger.info("Making predictions")
    # y_pred = model(X_val)
    y_pred = model.predict(X_val)

    # Evaluate the model
    # Use detached numpy arrays
    mse = mean_squared_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    logger.info(f"Root mean squared error: {rmse}")

    # Make predictions on the test set
    # y_pred_test = model(X_test)
    y_pred_test = model.predict(X_test)

    # Save the predictions using cols "Id" and "SalePrice" and include timestamp
    logger.info("Saving predictions")
    test_data["SalePrice"] = y_pred_test
    test_data[["Id", "SalePrice"]].to_csv(f"../submissions/{args.architecture}_{args.model}_{time.strftime('%Y_%m_%d_%H:%M:%S')}.csv", index=False)

    # Save the model
    logger.info("Saving model")
    model_path = f"models/{args.architecture}_{args.model}_{time.strftime('%Y_%m_%d_%H:%M:%S')}.pkl" 
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")

    # Save logs
    logger.info("Saving logs")
    log_path = f"logs/{args.architecture}_{args.model}_{time.strftime('%Y_%m_%d_%H:%M:%S')}.log"
    with open(log_path, "w") as log_file:
        log_file.write(f"Root mean squared error: {mse}")


if __name__ == "__main__":
    main()

