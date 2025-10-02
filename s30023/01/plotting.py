import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('data.csv')

plt.hist(df[' genre'])
plt.title('Songs in dataset by genre')
plt.xlabel('genre')
plt.ylabel('Amount of songs')
plt.show()
plt.savefig("test_histogram.png")
