import pandas as pd

def CleanData():
  df = pd.read_csv("car_sale_data.csv")

  df = df.drop_duplicates()

  df = df.dropna(subset=["price"])

# Simple fill for missing data
  df = df.fillna(method="ffill")

# Reset index
  df = df.reset_index(drop=True)

  return df


