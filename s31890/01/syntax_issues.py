import pandas as pd
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(
    prog="histogram_generator", description="Process CSV file and generate histogram"
)

parser.add_argument("file_path", help="Path to the CSV file")
parser.add_argument("column_name", help="Name of the column to generate histogram for")
parser.add_argument("min_value", type=int, help="Minimum value for the range")
parser.add_argument("max_value", type=int, help="Maximum value for the range")

args = parser.parse_args()

file_path = args.file_path
nazwa_kolumny = args.column_name
min_wartosc = args.min_value
max_wartosc = args.max_value

data = pd.read_csv(file_path)
df = pd.DataFrame(data)

rows_of_interest = df.iloc[min_wartosc:max_wartosc]
col_of_interest = rows_of_interest[nazwa_kolumny]

plt.hist(col_of_interest)
plt.title(f"Histogram {nazwa_kolumny}")
plt.savefig("histogram.png")
