import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')
print(df.head(5))


plt.hist(df["subscription_type"], bins=4, edgecolor='black')
plt.title("Number of subscriptions by type")
plt.xlabel("Subscription Type")
plt.ylabel("Number")
plt.savefig("plot.png")
plt.close()