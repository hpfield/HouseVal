import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
# ada boost regressor
from sklearn.ensemble import AdaBoostRegressor
# gradient boost regressor
from sklearn.ensemble import GradientBoostingRegressor
# xgboost regressor
from xgboost import XGBRegressor
import time

# Load datasets
train_df = pd.read_csv('data/train_cleaned.csv')
test_df = pd.read_csv('data/test_cleaned.csv')

# Save the Id column from test set for later use
test_ids = test_df['Id']

# Remove the 'ID' column
train_df.drop(columns=['Id'], inplace=True)
test_df.drop(columns=['Id'], inplace=True)

# Assuming the target column is named 'target' in the training set
X = train_df.drop(columns=['SalePrice'])
y = train_df['SalePrice']

# Standardise the data
scaler = StandardScaler()
X = scaler.fit_transform(X)
test_df = scaler.transform(test_df)

# Define models to try
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'RandomForestRegressor': RandomForestRegressor(),
    "ada_boost": AdaBoostRegressor(),
    "gradient_boost": GradientBoostingRegressor(),
    "xgboost": XGBRegressor()
}

# Fit models and make predictions on the test set
for name, model in models.items():
    model.fit(X, y)
    predictions = model.predict(test_df)

    # Save predictions to CSV
    output_df = pd.DataFrame({'Id': test_ids, 'SalePrice': predictions})
    output_file_name = f"../submissions/predictions_{name}_{time.strftime('%Y_%m_%d_%H:%M:%S')}.csv"
    output_df.to_csv(output_file_name, index=False)
    print(f'Predictions saved to {output_file_name}')


