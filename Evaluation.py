import matplotlib.pyplot as plt
from Processing import CleanData
# These two mode is to show any outliers and trends to know before we start training the data
df = CleanData()
# Histogram Model
df["Price"].hist(bins=50)
plt.title("Price Distribution")
plt.show()

# Scatter Plot
plt.scatter(df["Mileage"], df["Price"], alpha=0.3)
plt.xlabel("Mileage")
plt.ylabel("Price")
plt.title("Mileage vs Price")
plt.show()  