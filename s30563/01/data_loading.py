import pandas as pd

df = pd.read_csv("youtube-top-100-songs-2025.csv")

pd.set_option("display.max_columns", None)
print(df.head())
print(type(df))