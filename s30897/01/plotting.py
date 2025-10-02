import matplotlib.pyplot as plt
import kagglehub
import pandas as pd

path = kagglehub.dataset_download("ayeshaimran123/top-youtube-music-hits-2025")

df = pd.read_csv(path + "/youtube-top-100-songs-2025.csv")

print(df.columns)
df["view_count"] = df["view_count"].astype(str).str.replace(",", "").astype(int)

top10 = df.sort_values(by="view_count", ascending=False).head(10)

# Wykres
plt.figure(figsize=(12, 6))
plt.barh(top10["title"], top10["view_count"], color='skyblue')
plt.xlabel("Liczba wyświetleń (Views)")
plt.title("Top 10 najpopularniejszych piosenek YouTube")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()
