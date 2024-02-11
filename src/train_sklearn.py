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

# Import range of sklearn regression models
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# ada boost regressor
from sklearn.ensemble import AdaBoostRegressor
# gradient boost regressor
from sklearn.ensemble import GradientBoostingRegressor
# xgboost regressor
from xgboost import XGBRegressor


src_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(src_dir)


def predict_and_save(model, X_test, test, logger, rmse, model_name):
    # Make predictions on the test set
    # y_pred_test = model(X_test)
    y_pred_test = model.predict(X_test)

    # Save the predictions using cols "Id" and "SalePrice" and include timestamp
    logger.info("Saving predictions")
    test_data = test.copy()
    test_data["SalePrice"] = y_pred_test
    test_data[["Id", "SalePrice"]].to_csv(f"../submissions/{model_name}_{time.strftime('%Y_%m_%d_%H:%M:%S')}.csv", index=False)

    # Save the model
    logger.info("Saving model")
    model_path = f"models/sklearn/{model_name}_{time.strftime('%Y_%m_%d_%H:%M:%S')}.pkl" 
    joblib.dump(model, model_path)

    logger.info(f"Model saved to {model_path}")

def train(x_train, y_train, logger):
    # Train on a range of sklearn models
    models = {
        "linear": {"model":  LinearRegression()},
        "mlp": {"model":  MLPRegressor()},
        "random_forest": {"model": RandomForestRegressor()},
        "svr": {"model": SVR()},
        "ada_boost": {"model": AdaBoostRegressor()},
        "gradient_boost": {"model":  GradientBoostingRegressor()},
        "xgboost": {"model": XGBRegressor()}
    }

    # Train the models
    for model_name, model in models.items():
        logger.info(f"Training {model_name}")
        model["model"].fit(x_train, y_train)
        logger.info(f"Training {model_name} complete")

    # Return models in dict
    return models

def evaluate(models, X_val, y_val, logger):
    # Evaluate the models
    for model_name, model in models.items():
        logger.info(f"Evaluating {model_name}")
        y_pred = model['model'].predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        logger.info(f"Root mean squared error for {model_name}: {rmse}")
        model['rmse'] = rmse


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load the training data
    logger.info(f"Loading data from train_cleaned.csv")
    data = pd.read_csv("data/" + "train_cleaned.csv")

    # Split the data into features and target
    X = data.drop("SalePrice", axis=1)
    y = data["SalePrice"]

    # Load the test data
    logger.info(f"Loading data from test_cleaned.csv")
    test_data = pd.read_csv("data/" + "test_cleaned.csv")

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

    models = train(X_train, y_train, logger)
    evaluate(models, X_val, y_val, logger)

    # Save the best model
    best_model = min(models, key=lambda k: models[k]['rmse'])
    logger.info(f"Best model: {best_model}")
    model = models[best_model]
    predict_and_save(model['model'], X_test, test_data, logger, model['rmse'], best_model)
    

    # Save logs
    logger.info("Saving logs")
    log_path = f"logs/sklearn_{time.strftime('%Y_%m_%d_%H:%M:%S')}.log"
    with open(log_path, "w") as log_file:
        for model_name, model in models.items():
            log_file.write(f"{model_name}: {model['rmse']}\n")

    


if __name__ == "__main__":
    main()

