# https://www.kaggle.com/datasets/justinas/nba-players-data?resource=download
import pandas as pd

path = "all_seasons.csv"

df = pd.read_csv(path)

print(df.head())