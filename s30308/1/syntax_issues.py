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

filtered_col = df.query(f'{col} > {args.min_wartosc} and {col} < {args.max_wartosc}')
#filtered_col = df[df[col] > args.min_wartosc]
df = pd.DataFrame(filtered_col)

df.hist()
plt.show()