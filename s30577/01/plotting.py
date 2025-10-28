import matplotlib.pyplot as plt
import pandas as pd

csv = "/Users/computer/Library/Mobile Documents/com~apple~CloudDocs/STUDIA/5 Semestr/youtube-top-100-songs-2025.csv"

df = pd.read_csv(csv)
print(df.columns)

plt.figure(figsize=(12, 6))
plt.bar(df['duration'][:5], df['title'][:5])
plt.title('Przykładowy wykres')
plt.xlabel('czas trwania')
plt.ylabel('tytuly')
plt.tight_layout()
plt.savefig("wykres_top5yt.png")
plt.show()
