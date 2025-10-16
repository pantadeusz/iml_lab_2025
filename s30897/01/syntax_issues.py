import sys
import pandas as pd
import matplotlib.pyplot as plt

if len(sys.argv) != 5:
    print("Proszę podać tylko 5 argumentów: <ścieżka> <nazwa> <min. wartość> <max. wartość>")
    sys.exit(1)

file_path = sys.argv[1]
stat_name = sys.argv[2]
min_value = int(sys.argv[3])
max_value = int(sys.argv[4])

try:
    df = pd.read_csv(file_path)
except Exception as e:
    print("Błąd przy wczytywaniu pliku CSV: ", e)
    sys.exit(1)

df[stat_name] = pd.to_numeric(df[stat_name])

filtered = df[(df[stat_name] >= min_value) & (df[stat_name] <= max_value)]

print("Wartości w danym zakresie: \n", filtered)

plt.figure(figsize=(16, 6))
plt.bar(filtered["Title"], filtered[stat_name], color="skyblue", edgecolor="black")
plt.xticks(rotation=45, ha="right")
plt.xlabel("Tytuł piosenki")
plt.ylabel("Liczba wyświetleń")
plt.title(f"Wykres piosenek wg '{stat_name}' ({min_value}–{max_value})")

# Dodanie wartości na słupkach
for idx, val in enumerate(filtered[stat_name]):
    plt.text(idx, val + (val*0.02), f"{val:,}", ha="center")

plt.tight_layout()
plt.savefig("histogram.png")
print("Histogram zapisany do pliku histogram.png")
