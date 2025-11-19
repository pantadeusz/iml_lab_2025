import sys
import pandas as pd
import matplotlib.pyplot as plt

CSV_path = sys.argv[1]
col_title = sys.argv[2]
min_val = float(sys.argv[3])
max_val = float(sys.argv[4])

df = pd.read_csv(CSV_path)
filtered_Data = df[(df[col_title] >= min_val) & (df[col_title] <= max_val)]
print(filtered_Data)

plt.hist(filtered_Data[col_title])
plt.title(f"Histogram wartości z kolumny: {col_title}, z zakresu {min_val} - {max_val}")
plt.xlabel(col_title)
plt.ylabel("Liczba wystąpień")
plt.savefig("kawa.png")
plt.show()