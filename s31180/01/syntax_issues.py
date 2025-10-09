# https://www.kaggle.com/datasets/justinas/nba-players-data?resource=download

import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 5:
    print("Użycie: python syntax_issues.py <plik.csv> <kolumna> <min> <max>")
    sys.exit(1)

path = sys.argv[1]
column_name = sys.argv[2]
min_val = float(sys.argv[3])
max_val = float(sys.argv[4])

df = pd.read_csv(path)

if column_name not in df.columns:
    print(f"Brak kolumny '{column_name}'")
    sys.exit(1)

filtered = df[(df[column_name] >= min_val) & (df[column_name] <= max_val)][column_name]

print(filtered.values)

plt.hist(filtered, bins=20, color='blue', edgecolor='black')
plt.title(f"Histogram '{column_name}' w zakresie {min_val} - {max_val}")
plt.xlabel(column_name)
plt.ylabel("Ilość")
plt.savefig("histogram.png")
plt.show()
