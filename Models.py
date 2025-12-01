import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
<<<<<<< HEAD
import sklearn as metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from Processing import CleanData

# Linear Regression Model
def multipleLinearRegression(df):
=======
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import sklearn as metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Processing import CleanData

def multipleLinearRegression(df = CleanData().copy()):
    
    if 'Year' in df.columns:
    # Calculate age by subtracting the manufacturing year from the current year
        df['Car_Age'] = 2025 - df['Year']
    
    # Drop the original 'Year' column since 'Car_Age' is more descriptive for depreciation
        df.drop('Year', axis=1, inplace=True)
    df['Price_LOG'] = np.log(df['Price'])
# Identify categorical columns for one-hot encoding
    categorical_cols = ['Manufacture','Model', 'Fuel type']
>>>>>>> 806346762ec1e90db1470df9ad768424a636a837

    df = df.rename(columns={
        "Manufacturer": "manufacturer",
        "Model": "model",
        "Engine size": "engine_size",
        "Fuel type": "fuel_type",
        "Year of manufacture": "year",
        "Mileage": "mileage",
        "Price": "price"
    })

    df["car_age"] = 2025 - df["year"]

    df["price_log"] = np.log(df["price"])

    categorical_cols = ["manufacturer", "model", "fuel_type"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop(["price", "price_log"], axis=1)
    y = df_encoded["price_log"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)

<<<<<<< HEAD
    y_pred_log = reg_model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_actual = np.exp(y_test)

    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    metrics_dict = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2
    }

    results = pd.DataFrame({
        "Actual": y_actual,
        "Predicted": y_pred
    })

    return reg_model, metrics_dict, results



# Decision Tree Model
def decisionTree(test_size=0.3, random_state=100):

    DT = CleanData().copy()

    # Correct column name
    DT["Car_Age"] = datetime.now().year - DT["Year of manufacture"]

    # Drop original year column
    DT.drop("Year of manufacture", axis=1, inplace=True)

    # Correct categorical column names
    categorical_cols = ["Manufacturer", "Model", "Fuel type"]

    # One-hot encode
    df_encoded = pd.get_dummies(DT, columns=categorical_cols, drop_first=True)

    # Features and target
    X = df_encoded.drop("Price", axis=1)
    y = df_encoded["Price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Model
=======
    y_pred_test_log = reg_model.predict(X_test)
    y_test_pred = np.exp(y_pred_test_log)
    y_test_actual = np.exp(y_test)
    y_pred_train_log = reg_model.predict(X_train)
    y_train_pred = np.exp(y_pred_train_log)
    print("Prediction for test set for Linear Regression Model: {}".format(y_test_pred))
    reg_model_diff = pd.DataFrame({'Actual value': y_test_actual, 'Predicted value': y_test_pred})
    reg_model_diff
    mae = metrics.mean_absolute_error(y_test, y_test_pred)
    mse = metrics.mean_squared_error(y_test, y_test_pred)
    r2 = np.sqrt(metrics.mean_squared_error(y_test, y_test_pred))

    LinearRegression_metrics = {
        'MAE': mae, 'MSE': mse, 'RMSE': r2}

def decisionTree(test_size = 0.3, random_state = 100):
    DT = CleanData().copy()
    DT["Car_Age"] = datetime.now().year - DT["Year_of_manufacture"]
    DT.drop("Year_of_manufacture", axis=1, inplace=True)

    categorical_cols = ["Manufacturer", "Model", "Fuel_Type"]
    DT_encoded = pd.get_dummies(DT, columns=categorical_cols, drop_first=True)

    X = DT_encoded.drop("Price", axis=1)
    y = DT_encoded["Price"].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
>>>>>>> 806346762ec1e90db1470df9ad768424a636a837
    model = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
    )
<<<<<<< HEAD

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()

    training_time = (end - start) * 1000  # ms

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
=======
    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    training_time = (end - start) * 1000
    y_pred = model.predict(X_test)

>>>>>>> 806346762ec1e90db1470df9ad768424a636a837
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)
<<<<<<< HEAD

    metrics_dict = {
        "MAE": MAE,
        "MSE": MSE,
        "RMSE": RMSE,
        "R2": R2,
    }

=======
    metrics_dict = {"MAE": MAE,"MSE": MSE,"RMSE": RMSE,"R2": R2}
>>>>>>> 806346762ec1e90db1470df9ad768424a636a837
    results_DT = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    print("\nDecision Tree Regression Results:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")
<<<<<<< HEAD

    print(f"\nDecision Tree Training Time: {training_time:.3f} ms")

    return model, metrics_dict, results_DT

# Random Tree Model

def randomForest(test_size=0.3, random_state=100):

    # LOAD CLEANED DATA
    df = CleanData().copy()

    # Rename columns for easier handling
    df = df.rename(columns={
        "Manufacturer": "manufacturer",
        "Model": "model",
        "Engine size": "engine_size",
        "Fuel type": "fuel_type",
        "Year of manufacture": "year",
        "Mileage": "mileage",
        "Price": "price"
    })

    # Add car age (better than using year directly)
    df["car_age"] = datetime.now().year - df["year"]

    # One-hot encode categorical columns
    categorical_cols = ["manufacturer", "model", "fuel_type"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    # Features and target
    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Random Forest Model
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)

    metrics_dict = {
        "MAE": round(MAE, 4),
        "MSE": round(MSE, 4),
        "RMSE": round(RMSE, 4),
        "R2": round(R2, 4)
    }

    print("\nRandom Forest Regression Results:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v}")

    # Comparison dataframe
    results_RF = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    return model, metrics_dict, results_RF

# Gradient Boosting model
def gradientBoosting(test_size=0.3, random_state=100):

    df = CleanData().copy()

    df = df.rename(columns={
        "Manufacturer": "manufacturer",
        "Model": "model",
        "Engine size": "engine_size",
        "Fuel type": "fuel_type",
        "Year of manufacture": "year",
        "Mileage": "mileage",
        "Price": "price"
    })

    df["car_age"] = datetime.now().year - df["year"]

    categorical_cols = ["manufacturer", "model", "fuel_type"]
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]
=======
    print(f"\nDecision Tree Regression Training Time: {training_time:.3f} ms")

    return model, metrics_dict, results_DT, training_time

def GradientBoosting(test_size = 0.3, random_state = 100):
    GB = CleanData().copy()
    GB["Car_Age"] = datetime.now().year - GB["Year_of_manufacture"]
    GB.drop("Year_of_manufacture", axis=1, inplace=True)

    categorical_cols = ["Manufacturer", "Model", "Fuel_Type"]
    GB_encoded = pd.get_dummies(GB, columns=categorical_cols, drop_first=True)

    X = GB_encoded.drop("Price", axis=1)
    y = GB_encoded["Price"].values
>>>>>>> 806346762ec1e90db1470df9ad768424a636a837

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

<<<<<<< HEAD
    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
=======
    GB_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.9,
        random_state=random_state,
    )

    start = time.time()
    GB_model.fit(X_train, y_train)
    end = time.time()
    training_time = (end - start) * 1000
    y_pred = GB_model.predict(X_test)
>>>>>>> 806346762ec1e90db1470df9ad768424a636a837

    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = np.sqrt(MSE)
    R2 = r2_score(y_test, y_pred)
<<<<<<< HEAD

    metrics_dict = {
        "MAE": round(MAE, 4),
        "MSE": round(MSE, 4),
        "RMSE": round(RMSE, 4),
        "R2": round(R2, 4)
    }

    print("\nGradient Boosting Regression Results:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v}")

    results_GB = pd.DataFrame({
        "Actual": y_test,
        "Predicted": y_pred
    })

    return model, metrics_dict, results_GB

# User- Input prediction model
def predict_single_car(model, df_original,
                       manufacturer, model_name, year, mileage, fuel_type, engine_size):

    # Clean training dataset column names
    df = df_original.rename(columns={
        "Manufacturer": "manufacturer",
        "Model": "model",
        "Engine size": "engine_size",
        "Fuel type": "fuel_type",
        "Year of manufacture": "year",
        "Mileage": "mileage",
        "Price": "price"
    })

    df["car_age"] = datetime.now().year - df["year"]

    # Build user input dataframe
    new_car = pd.DataFrame([{
        "manufacturer": manufacturer,
        "model": model_name,
        "engine_size": engine_size,
        "fuel_type": fuel_type,
        "year": year,
        "mileage": mileage,
        "car_age": datetime.now().year - year
    }])

    # One-hot encode training data
    train_encoded = pd.get_dummies(df.drop("price", axis=1), drop_first=True)

    # One-hot encode user input
    new_car_encoded = pd.get_dummies(new_car, drop_first=True)

    # Align columns (this is the MOST important line)
    new_car_encoded = new_car_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    # Predict using the trained Gradient Boosting model
    X_new = new_car_encoded.values
    predicted_price = model.predict(X_new)[0]

    return predicted_price
=======
    metrics_dict = {"MAE": MAE, "MSE": MSE, "RMSE": RMSE, "R2": R2}
    results_GB = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    print("\nDecision Tree Regression Results:")
    for k, v in metrics_dict.items():
        print(f"{k}: {v:.4f}")
    print(f"\nDecision Tree Regression Training Time: {training_time:.3f} ms")

    return GB_model, metrics_dict, results_GB, training_time
>>>>>>> 806346762ec1e90db1470df9ad768424a636a837
