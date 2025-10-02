import pandas as pd
data =  pd.read_csv("youtube-top-100-songs-2025.csv")
df = pd.DataFrame(data)
print(df.head())