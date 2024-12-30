# HouseVal

**[Kaggle Competition Link](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview)**

This repository contains a complete end-to-end machine learning workflow aimed at predicting house sale prices given 79 explanatory variables from the Ames Housing dataset. The goal is to demonstrate proficiency in building scalable and reproducible machine learning pipelines using common industry practices.


## Project Overview

Predicting house prices is a classic regression problem frequently used to assess the effectiveness of various machine learning techniques. In this competition, we aim to estimate the final house sale price by leveraging data exploration, feature engineering, model experimentation, and hyperparameter tuning. The codebase in this repository highlights the typical steps involved in a simple data science project:



1. **Data Acquisition & Organisation**
2. **Exploratory Data Analysis (EDA)**
3. **Feature Engineering & Data Cleaning**
4. **Model Development** (including Linear Regression, Random Forests, Gradient Boosting, and Neural Networks)
5. **Evaluation & Submission** (using Root Mean Squared Error on the log of SalePrice)


## Repository Structure

```
.
├── data/
│   ├── raw/                  # Original Kaggle data (train/test)
│   ├── processed/            # Data after cleaning and feature engineering
├── notebooks/
│   ├── copy_data.ipynb       # Copies processed data into src/data
│   ├── eda.ipynb             # Exploratory data analysis (insights on distributions, correlations, etc.)
│   ├── feature_engineering.ipynb 
│   ├── initial_data_separation.ipynb
├── src/
│   ├── architectures/        # Neural network architectures (PyTorch)
│   ├── data/                 # Data (used by training scripts)
│   ├── logs/                 # Logging files
│   ├── models/               # Serialised trained models
│   ├── runs/                 # Checkpoints and run artifacts
│   ├── __init__.py
│   ├── train.py              # PyTorch-based training script
│   ├── train_predict.py      # Scikit-learn-based training & prediction workflow
│   ├── train_simple.py       # Simple scikit-learn training approach
│   └── train_sklearn.py      # Unified scikit-learn training pipeline
├── submissions/              # Model predictions to upload on Kaggle
├── concat.py                 # Utility script to concatenate or clean files
└── README.md                 # Project documentation
```


### Key Files & Directories

* \
  **notebooks/eda.ipynb** Performs exploratory data analysis to uncover relationships, distributions, missing data patterns, and potential outliers. Visualisations include histograms, box plots, pairplots, correlation heatmaps, and missing-value plots.
* \
  **notebooks/feature_engineering.ipynb** Demonstrates feature engineering workflows, including:
  * Handling categorical variables via one-hot-encoding or ordinal mappings
  * Combining multiple features (e.g., total square footage)
  * Creating new features (e.g., date sold, binary flags for porches/pools)
  * Imputation (median, mode) for missing data
* \
  **notebooks/initial_data_separation.ipynb** Establishes data separation steps, ensuring training and test sets are clearly demarcated and that relevant transformations are consistently applied. Also documents how columns with high missingness or irrelevance may be dropped.
* \
  **src/train.py** A PyTorch-based training script that sets up an MLP architecture, trains it with a configurable number of epochs and learning rate, and logs the results. Outputs final trained model and predictions for submission.
* \
  **src/train_sklearn.py** A scikit-learn-based training pipeline that systematically evaluates multiple regression models (Linear, Random Forest, Gradient Boosting, XGBoost, etc.). Uses random splits for validation, standardised features, and logs the RMSE to identify the best model.
* \
  **src/train_predict.py** An alternative scikit-learn workflow that shows how to quickly train multiple models (e.g., LinearRegression, RandomForestRegressor, AdaBoostRegressor, XGBRegressor) and generate Kaggle submission files.
* \
  **src/train_simple.py** A more lightweight approach for scikit-learn model testing, focusing on a smaller subset of models with minimal overhead (useful for quick experiments).


## Machine Learning Workflow Highlights




1. **Exploratory Data Analysis**
   * Visualised missing data, outliers, and key correlations (e.g., OverallQual, TotalSF).
   * Quantified relationships between categorical features and target variable (SalePrice).
2. **Feature Engineering**
   * Performed one-hot encoding for high-cardinality categorical variables.
   * Mapped ordinal features (e.g., quality/condition metrics) to numeric scales.
   * Created combined features such as `TotalSF` and binary indicators (`WoodDeckSF_Present`, etc.).
   * Used custom imputation strategies (median, mode) for missing values.
3. **Model Development**
   * Utilised various regression algorithms:
     * Linear Regression and Regularised Linear Models (Lasso, Ridge)
     * Tree-Based Models (Random Forest, GradientBoostingRegressor)
     * XGBoost for gradient boosting with flexible hyperparameter tuning
     * Neural Networks using a feed-forward MLP architecture (PyTorch)
   * Investigated hyperparameters (e.g., number of estimators in Random Forest, learning rate for MLP).
4. **Evaluation & Validation**
   * Leveraged train-validation splits for unbiased performance estimates.
   * Monitored Root Mean Squared Error (RMSE) to align with Kaggle’s evaluation metric.
5. **Reproducible Pipeline**
   * Scripts in `src/` illustrate how to load data, transform it consistently, train models, and export predictions.
   * Logging is used to record training progress and final performance.
   * Final output predictions are saved in `submissions/` with time-stamped filenames.


## Performance

Depending on the model, final Kaggle submissions reached an RMSE score of around 0.13–0.14 (as evaluated by the log-based metric on Kaggle). While not state-of-the-art, this performance is solid for a straightforward pipeline and demonstrates effective use of:

* Feature engineering best practices
* Ensembling methods (Random Forest, Gradient Boosting, XGBoost)
* Neural Networks for regression tasks


