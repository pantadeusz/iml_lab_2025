import pandas as pd
csv = "/Users/computer/.cache/kagglehub/datasets/ayeshasiddiqa123/top-100-trending-music-on-youtube/versions/1/youtube-top-100-songs-2025.csv"

data = pd.read_csv(csv)
print(data['title'].head())