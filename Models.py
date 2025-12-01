import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import sklearn.metrics as metrics
from sklearn.metrics import r2_score

def multipleLinearRegression(df: pd.DataFrame):
    
    if 'Year' in df.columns:
    # Calculate age by subtracting the manufacturing year from the current year
        df['Car_Age'] = 2025 - df['Year']
    
    # Drop the original 'Year' column since 'Car_Age' is more descriptive for depreciation
        df.drop('Year', axis=1, inplace=True)
    df['Price_LOG'] = np.log(df['Price'])
# Identify categorical columns for one-hot encoding
    categorical_cols = ['Manufacture','Model', 'Fuel type']

# Apply one-hot encoding to categorical columns
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Make list of the independent variable and dependent variable
# Ensure 'Price' is the dependent variable and others are independent
    X = df_encoded.drop(['Price', 'Price_LOG'], axis=1)
    y = df_encoded['Price_LOG'].values 
    feature_names = X.columns.tolist()
    X = X.values
    X_train,X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 100)
    y_train.shape
    y_test.shape
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
#Printing the model coefficients
    #print('Intercept: ',reg_model.intercept_)
# pair the feature names with the coefficients

    y_pred_test_log = reg_model.predict(X_test)
    y_test_pred = np.exp(y_pred_test_log)
    y_test_actual = np.exp(y_test)
    y_pred_train_log = reg_model.predict(X_train)
    y_train_pred = np.exp(y_pred_train_log)
    #print("Prediction for test set: {}".format(y_test_pred))
    reg_model_diff = pd.DataFrame({'Actual value': y_test_actual, 'Predicted value': y_test_pred})
    reg_model_diff
    mae = metrics.mean_absolute_error(y_test, y_test_pred)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    r2 = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

    #print('Mean Absolute Error:', mae)
    #print('Mean Square Error:', mse)
    #print('Root Mean Square Error:', r2)

