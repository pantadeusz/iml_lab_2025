import matplotlib.pyplot as plt
import pandas as pd

# https://www.kaggle.com/datasets/ayeshaimran123/top-youtube-music-hits-2025
df = pd.read_csv("youtube-top-100-songs-2025.csv")

df_top10 = df.sort_values(by="view_count", ascending=False).head(10)

plt.figure(figsize=(8,8))
plt.bar(df_top10["channel"], df_top10["view_count"] / 1_000_000)
plt.title("Top 10 piosenek – liczba wyświetleń (mln)")
plt.xlabel("Kanał")
plt.ylabel("Liczba wyświetleń [mln]")
plt.xticks(rotation=20)
plt.savefig("top10_yt_songs.png")

plt.show()

