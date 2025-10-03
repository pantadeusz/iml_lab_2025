import pandas as pd


def load_data_to_df(data_set: str) -> pd.DataFrame:
    data = pd.read_csv(data_set)
    return pd.DataFrame(data)


def main():
    df = load_data_to_df("flight_data_2024.csv")
    print(df.head())


if __name__ == "__main__":
    main()
