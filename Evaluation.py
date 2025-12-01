import matplotlib.pyplot as plt
from Processing import CleanData

def ModelTrends(df):
    # Histogram
    plt.figure(figsize=(8,5))
    df["Price"].hist(bins=50)
    plt.title("Price Distribution")
    plt.savefig("price_distribution.png")
    plt.close()

    # Scatter Plot
    plt.figure(figsize=(8,5))
    plt.scatter(df["Mileage"], df["Price"], alpha=0.3)
    plt.xlabel("Mileage")
    plt.ylabel("Price")
    plt.title("Mileage vs Price")
    plt.savefig("mileage_vs_price.png")
    plt.close()
