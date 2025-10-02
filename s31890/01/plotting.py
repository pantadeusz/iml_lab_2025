import pandas as pd
import kagglehub

path = kagglehub.dataset_download(
    "wardabilal/exploring-coffee-sales-with-eda-and-visualization"
)

file_path = f"{path}/Coffe_sales.csv"
print("Path to dataset files:", file_path)

data = pd.read_csv(file_path)
df = pd.DataFrame(data)

import matplotlib.pyplot as plt

plt.bar(df["Month_name"], df["money"])
plt.title("Coffee Sales by Month")
plt.xlabel("Month")
plt.ylabel("Amount Spent")
plt.savefig("coffee_sales_plot.png")
