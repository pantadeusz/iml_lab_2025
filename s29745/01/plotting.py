import matplotlib.pyplot as plt
import data_loading as dl


def main():
    df = dl.load_data_to_df("flight_data_2024.csv")

    delayed_flights = df[df["arr_delay"] > 0].groupby("month").size()

    plt.bar(delayed_flights.index, delayed_flights.values)
    plt.title("Liczba opóźnionych lotów w danym miesiącu (2024)")
    plt.xlabel("Miesiąc")
    plt.xticks(delayed_flights.index)
    plt.ylabel("Liczba lotów")
    plt.grid(True, alpha=0.3)
    plt.show()


if __name__ == "__main__":
    main()
