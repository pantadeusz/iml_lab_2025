import pandas as pd
# https://www.kaggle.com/datasets/ayeshaimran123/top-youtube-music-hits-2025
df = pd.read_csv("youtube-top-100-songs-2025.csv")

pd.set_option("display.max_columns", None)
print(df.head())
print(type(df))