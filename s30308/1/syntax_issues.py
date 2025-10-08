import argparse
import pandas as pd
from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("sciezka_do_pliku_csv")
parser.add_argument("nazwa_kolumny")
parser.add_argument("min_wartosc")
parser.add_argument("max_wartosc")

args = parser.parse_args()

df = pd.read_csv(args.sciezka_do_pliku_csv)
col = args.nazwa_kolumny

filtered_df = df.query(f"{col} > {args.min_wartosc} and {col} < {args.max_wartosc}")

filtered_df[col].hist()
plt.title(f"Histogram kolumny '{col}'")
plt.xlabel(col)
plt.ylabel("Liczba wystąpień")

plt.savefig("histogram.png")