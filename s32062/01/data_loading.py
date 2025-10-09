import pandas as pd

# Przykładowe dane
data = {"kolumna1": [1, 2, 3], "kolumna2": [4, 5, 6]}
df = pd.DataFrame(data)
print(df[0])

df1 = pd.read_csv("lung_cancer_dataset.csv")
print(df1.head())
