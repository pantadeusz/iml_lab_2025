import pandas as pd

file_path = "netflix_titles.csv"

df = pd.read_csv(file_path)

print(df.head())
