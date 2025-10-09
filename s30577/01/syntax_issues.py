import pandas as pd
import matplotlib.pyplot as plt
import sys


if len(sys.argv) != 5:
    print("użycie: python syntax_issues.py <ścieżka_do_pliku_csv> <nazwa_kolumny> <min_wartość> <max_wartość>")


csv = sys.argv[1]
kolumna = sys.argv[2]
min = int(sys.argv[3])
max = int(sys.argv[4])

df = pd.read_csv(csv)

filtrowanie = df[(df[kolumna] >= min) & (df[kolumna] <= max)]

plt.hist(filtrowanie[kolumna], bins=20)
plt.title('Histogram dla kolumny ' + kolumna)

plt.savefig("histogram.png")

print(filtrowanie[['title', kolumna]])