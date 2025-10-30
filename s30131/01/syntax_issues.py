import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("file_path")
parser.add_argument("column")
parser.add_argument("min", type=float)
parser.add_argument("max", type=float)
arguments = parser.parse_args()
df = pd.read_csv(arguments.file_path)

filtered = df[(df[arguments.column] > arguments.min) & (df[arguments.column] < arguments.max)]

print(filtered[arguments.column])

plt.hist(filtered[arguments.column]); plt.xlabel("x"); plt.ylabel("y"); plt.savefig("histogram.png")



