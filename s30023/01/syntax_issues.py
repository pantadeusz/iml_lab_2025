import sys
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    if len(sys.argv) != 5:
        print('Usage: python syntax_issues.py <file.csv> <column name> <min> <max>')
        sys.exit(1)

    filename = sys.argv[1]
    column_name = sys.argv[2]
    min = int(sys.argv[3])
    max = int(sys.argv[4])

    df = pd.read_csv(filename)
    df = df[df[column_name].between(min, max)]

    data_for_plot = df.groupby('Year')[column_name].mean()

    plt.plot(data_for_plot.index, data_for_plot.values)
    plt.xlabel('Year')
    plt.ylabel(column_name)
    plt.savefig("hist_8.png")
