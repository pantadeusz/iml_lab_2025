import pandas as pd

path = './train.csv'
df = pd.read_csv(path)
print(df.head())
