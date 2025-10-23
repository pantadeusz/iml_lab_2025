import sys

import matplotlib.pyplot as plt
import pandas as pd

def main():
    _, path, column, min_value, max_value = sys.argv

    df = pd.read_csv(path)
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df = df.dropna(subset=[column])

    df_filtered = df[(df[column] > float(min_value)) & (df[column] < float(max_value))]
    plt.hist(df_filtered[column])
    plt.title("Histogram")
    plt.savefig("histogram.png")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 5:
        main()
    else:
        print("Niepoprawne dane")
