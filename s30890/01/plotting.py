import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('dataset.csv')
df.columns = ['f1','f2','f3','f4','f5','label']

plt.figure(figsize = (10,10))
plt.scatter(df['f1'],df['f5'],c=df['label'].astype("category").cat.codes,cmap='rainbow')

plt.xlabel('alert')
plt.ylabel('magnitude')
plt.show()