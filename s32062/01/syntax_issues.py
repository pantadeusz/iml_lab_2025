import argparse
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Process some inputs.")

parser.add_argument(
    "csv_path", help="Path to your csv file containing data for the model"
)
parser.add_argument("column_name", help="Your age")
parser.add_argument("min", type=int, help="Min value")
parser.add_argument("max", type=int, help="Max value")

args = parser.parse_args()

df = pd.read_csv(args.csv_path)

filtered_df = df[
    (df[args.column_name] >= args.min) & (df[args.column_name] <= args.max)
]

print(filtered_df[args.column_name])

plt.hist(filtered_df[args.column_name], bins=30, color="skyblue", edgecolor="black")
plt.xlabel(args.column_name)
plt.ylabel("count")
plt.title("User age")
plt.show()
plt.savefig("syntax_issues-plot.png")
