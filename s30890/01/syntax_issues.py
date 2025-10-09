import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 5:
    print("python syntax_issues.py <ścieżka_do_pliku_csv> <nazwa_kolumny> <min_wartość> <max_wartość>")
    sys.exit(1)

csv_path = sys.argv[1]
column_name = sys.argv[2]
min_val = float(sys.argv[3])
max_val = float(sys.argv[4])

df = pd.read_csv(csv_path)

if column_name not in df.columns:
    print(f"blad: kolumna '{column_name}' nie istnieje w pliku csv")
    sys.exit(1)

filtered = df[(df[column_name] >= min_val) & (df[column_name] <= max_val)]

print(f"wartosci kolumny '{column_name}' w zakresie {min_val}–{max_val}:")
print(filtered[column_name])

plt.hist(filtered[column_name], bins=20, edgecolor='black')
plt.title(f'Histogram wartości z kolumny {column_name}')
plt.xlabel(column_name)
plt.ylabel('Liczba wystąpień')

plt.savefig('histogram.png')
plt.show()
