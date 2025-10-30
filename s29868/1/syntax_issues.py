import pandas as pd
import matplotlib.pyplot as plt
import sys

if len(sys.argv) < 5:
    print("Usage: python syntax_issues.py <ścieżka_do_pliku_csv> <nazwa_kolumny> <min_wartość> <max_wartość>")
    sys.exit(1)

file_path = sys.argv[1]
column_name = sys.argv[2]
min_value = int(sys.argv[3])
max_value = int(sys.argv[4])


df = pd.read_csv(file_path)
df.columns = df.columns.str.strip()


filtered = df[ (df[column_name] >= min_value) & (df[column_name] <= max_value)]




plt.hist(filtered[column_name], bins=10, edgecolor="black")
plt.title(f"Histogram of '{column_name}' ({min_value} - {max_value})")
plt.xlabel(column_name)
plt.ylabel("Number of rows")

plt.savefig("histogram.png")
plt.close()
