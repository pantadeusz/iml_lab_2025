import pandas as pd
csv = "/Users/computer/Library/Mobile Documents/com~apple~CloudDocs/STUDIA/5 Semestr/youtube-top-100-songs-2025.csv"

data = pd.read_csv(csv)
print(data['title'].head())