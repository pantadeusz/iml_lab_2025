import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="filtrowanie")
parser.add_argument("file_path")
parser.add_argument("column")
parser.add_argument("min_value", type=float)
parser.add_argument("max_value", type=float)

args = parser.parse_args()

df = pd.read_csv(args.file_path)

if args.column not in df.columns:
    print(f"Column not found: {args.column}")
    exit(1)

filtered = df[(df[args.column] > args.min_value) & (df[args.column] < args.max_value)]

print(filtered[args.column])

plt.hist(filtered[args.column])
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("hist.png")
