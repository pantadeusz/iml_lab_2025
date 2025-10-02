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

money_average = df["money"].rolling(window=20).mean()

plt.plot(pd.to_datetime(df["Date"]), money_average)
plt.title("Coffee Sales Average by Date")
plt.xlabel("Date")
plt.ylabel("Running Average Amount Spent")
plt.savefig("coffee_sales_plot.png")
