import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import numpy as np

# Load datasets
train_df = pd.read_csv('data/train_cleaned.csv')
test_df = pd.read_csv('data/test_cleaned.csv')

# Remove the 'ID' column
train_df.drop(columns=['Id'], inplace=True)
test_df.drop(columns=['Id'], inplace=True)

# Assuming the target column is named 'target'
X = train_df.drop(columns=['SalePrice'])
y = train_df['SalePrice']

# Create train-validation split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to try
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForestRegressor': RandomForestRegressor()
}

# Iterate over models, training and evaluating them
results = {}
for name, model in models.items():
    pipeline = make_pipeline(StandardScaler(), model)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_val)
    mse = mean_squared_error(y_val, predictions)
    results[name] = mse
    print(f'{name}: Mean Squared Error = {mse}')

# Display results
print("Model performance on validation set:")
for model, mse in results.items():
    print(f"{model}: RMSE = {np.sqrt(mse)}")

