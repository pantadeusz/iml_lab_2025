import pandas as pd
import matplotlib.pyplot as plt 
import data_loading as dl
import sys
import os

def main():
    csv_path, col_name, min_val, max_val = check_for_args()

    df = dl.load_data_to_df(csv_path)

    if col_name not in df.columns:
        print(f"Kolumna {col_name} nie istnieje")
        sys.exit(1)

    try:
        df[col_name] = pd.to_numeric(df[col_name], errors="coerce")
    except Exception:
        print("Kolumna nie jest numeryczna")
        sys.exit(1)

    filtered_data = df[df[col_name].between(min_val, max_val)]

    print(filtered_data[col_name])
    plot_and_save_to_pdf(filtered_data, col_name, min_val, max_val)

    

def plot_and_save_to_pdf(filtered_data: pd.DataFrame, col_name: str, min_val: float, max_val: float):
    plt.hist(filtered_data[col_name].dropna(), edgecolor="black")
    plt.xlabel(col_name)
    plt.ylabel("Liczba")
    plt.title(f"Histogram wartości kolumny '{col_name}' ({min_val}–{max_val})")
    plt.savefig("histogram.png")
    plt.close()


def check_for_args():
    if len(sys.argv) != 5:
        print("Niepoprawna liczba argumentow")
        print("Uzycie: python3 syntax_issues.py <ścieżka_do_pliku_csv> <nazwa_kolumny> <min_wartość> <max_wartość>")
        sys.exit(1)

    if not os.path.exists(sys.argv[1]):
        print("Sciezka do pliku csv jest niepoprawna")
        sys.exit(1)

    csv_path = sys.argv[1]
    col_name = sys.argv[2]

    try:
        min_val = float(sys.argv[3])
        max_val = float(sys.argv[4])
    except ValueError:
        print("min i max musza byc liczbami")
        sys.exit(1)

    return csv_path, col_name, min_val, max_val


if __name__ == "__main__":
    main()
