# https://www.kaggle.com/datasets/justinas/nba-players-data?resource=download
import pandas as pd
import matplotlib.pyplot as plt

path = "all_seasons.csv"

df = pd.read_csv(path)

df_mean = df.groupby('season')['pts'].mean().reset_index()

plt.plot(df_mean['season'], df_mean['pts'], marker='o')
plt.title('Punkty na sezon')
plt.xlabel('Sezon')
plt.ylabel('Średnia punktów')
plt.xticks(rotation=45)

plt.savefig("wykres.png")
plt.show()