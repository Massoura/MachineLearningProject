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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from Processing import CleanData


# ===========================
#  LINEAR REGRESSION MODEL
# ===========================
def multipleLinearRegression(df):

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

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_log = model.predict(X_test)
    y_pred = np.exp(y_pred_log)
    y_actual = np.exp(y_test)

    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_actual, y_pred)

    metrics_dict = {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    results = pd.DataFrame({"Actual": y_actual, "Predicted": y_pred})

    return model, metrics_dict, results



# ===========================
#  DECISION TREE REGRESSOR
# ===========================
def decisionTree(test_size=0.3, random_state=100):

    dt = CleanData().copy()

    dt["car_age"] = datetime.now().year - dt["Year of manufacture"]
    dt.drop("Year of manufacture", axis=1, inplace=True)

    categorical_cols = ["Manufacturer", "Model", "Fuel type"]
    df_encoded = pd.get_dummies(dt, columns=categorical_cols, drop_first=True)

    X = df_encoded.drop("Price", axis=1)
    y = df_encoded["Price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = DecisionTreeRegressor(
        criterion="squared_error",
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=random_state,
    )

    start = time.time()
    model.fit(X_train, y_train)
    end = time.time()
    training_time = (end - start) * 1000

    y_pred = model.predict(X_test)

    metrics_dict = {
        "MAE": mean_absolute_error(y_test, y_pred),
        "MSE": mean_squared_error(y_test, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
        "R2": r2_score(y_test, y_pred),
    }

    results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    return model, metrics_dict, results



# ===========================
#     RANDOM FOREST MODEL
# ===========================
def randomForest(test_size=0.3, random_state=100):

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

    df_encoded = pd.get_dummies(df, columns=["manufacturer", "model", "fuel_type"], drop_first=True)

    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics_dict = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "MSE": round(mean_squared_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2": round(r2_score(y_test, y_pred), 4)
    }

    results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    return model, metrics_dict, results



# ===========================
# GRADIENT BOOSTING REGRESSOR
# ===========================
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

    df_encoded = pd.get_dummies(df, columns=["manufacturer", "model", "fuel_type"], drop_first=True)

    X = df_encoded.drop("price", axis=1)
    y = df_encoded["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

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

    metrics_dict = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        "MSE": round(mean_squared_error(y_test, y_pred), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
        "R2": round(r2_score(y_test, y_pred), 4),
    }

    results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})

    return model, metrics_dict, results



# ===========================
# USER INPUT PREDICTION
# ===========================
def predict_single_car(model, df_original,
                       manufacturer, model_name, year, mileage, fuel_type, engine_size):

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

    new_car = pd.DataFrame([{
        "manufacturer": manufacturer,
        "model": model_name,
        "engine_size": engine_size,
        "fuel_type": fuel_type,
        "year": year,
        "mileage": mileage,
        "car_age": datetime.now().year - year
    }])

    train_encoded = pd.get_dummies(df.drop("price", axis=1), drop_first=True)
    new_car_encoded = pd.get_dummies(new_car, drop_first=True)

    new_car_encoded = new_car_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    X_new = new_car_encoded.values
    predicted_price = model.predict(X_new)[0]

    return predicted_price
