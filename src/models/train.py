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
import sklearn as sk
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


# Add the src directory for imports
src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(src_dir)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="linear", help="Model to train")
    parser.add_argument("--train_data", type=str, default="train_one-hot-encoded_no-missing_vals_mean_imputed.csv", help="Data to train on")
    parser.add_argument("--test_data", type=str, default="test_one-hot-encoded_no-missing_vals_mean_imputed.csv", help="Data to test on")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train for")
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
    data = pd.read_csv("../data/" + args.train_data)

    # Split the data into features and target
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    print("shape of X", X.shape)

    # Load the test data
    logger.info(f"Loading data from {args.test_data}")
    test_data = pd.read_csv("../data/" + args.test_data)

    print("shape of test_data", test_data.shape)
    X_test = test_data

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

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

    # Create architecture using sklearn
    if args.model == "linear":
        model = LinearRegression()
    elif args.model == "mlp":
        model = MLPRegressor(hidden_layer_sizes=(100, 100), max_iter=args.epochs, batch_size=args.batch_size, learning_rate_init=args.learning_rate)
    elif args.model == "random_forest":
        model = RandomForestRegressor(n_estimators=100, max_depth=10)
    elif args.model == "svr":
        model = SVR(kernel="rbf", C=100, gamma=0.1, epsilon=.1)

    # Train the model
    logger.info("Training model")
    model.fit(X_train, y_train)

    # Make predictions
    logger.info("Making predictions")
    y_pred = model.predict(X_val)

    # Evaluate the model
    mse = mean_squared_error(y_val, y_pred)
    logger.info(f"Mean squared error: {mse}")

    # Make predictions on the test set
    y_pred_test = model.predict(X_test)

    # Save the predictions using cols "Id" and "SalePrice" adn include timestamp
    logger.info("Saving predictions")
    test_data["SalePrice"] = y_pred_test
    test_data[["Id", "SalePrice"]].to_csv(f"../../submissions/{args.architecture}_{args.model}_{time.strftime('%Y%m%d%H%M%S')}.csv", index=False)

    # Save the model
    logger.info("Saving model")
    model_path = f"../models/{args.architecture}_{args.model}_{time.strftime('%Y%m%d%H%M%S')}.pkl" 
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")


if __name__ == "__main__":
    main()

