from Processing import CleanData
from Models import multipleLinearRegression, decisionTree, randomForest, gradientBoosting, predict_single_car
from Evaluation import ModelTrends
import pandas as pd

def run_all_models(df):

    lin_model, lin_metrics, lin_results = multipleLinearRegression(df)
    dt_model, dt_metrics, dt_results = decisionTree()
    rf_model, rf_metrics, rf_results = randomForest()
    gb_model, gb_metrics, gb_results = gradientBoosting()

    metrics_df = pd.DataFrame({
        "Linear Regression": lin_metrics,
        "Decision Tree": dt_metrics,
        "Random Forest": rf_metrics,
        "Gradient Boosting": gb_metrics
    }).T

    print("\n\n=== MODEL COMPARISON TABLE ===\n")
    print(metrics_df)

    return metrics_df


df_clean = CleanData()

model, metrics, results = randomForest()

ModelTrends(df_clean)

run_all_models(df_clean)

print("\n--- Enter Car Details for Prediction ---")
manufacturer = input("Manufacturer: ")
model_name = input("Model: ")
year = int(input("Year of manufacture: "))
mileage = int(input("Mileage: "))
fuel_type = input("Fuel type (e.g., Petrol, Diesel): ")
engine_size = float(input("Engine size (e.g., 1.6): "))

# Find average price in dataset
avg_data_price = df_clean["Price"].mean()

# Decide a "real-world" target mean (e.g., 15,000)
target_mean_price = 15000

# Compute scaling factor
scale_factor = target_mean_price / avg_data_price

raw_prediction = predict_single_car(model, df_clean,
                                manufacturer, model_name, year, mileage,
                                fuel_type, engine_size)
pred_price = raw_prediction * scale_factor

print(f"\nPredicted Resale Value: ${pred_price:,.2f}")
