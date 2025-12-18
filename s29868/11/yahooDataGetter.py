import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler


class YahooDataGetter:
    def __init__(self, ticker, start_date='2022-01-01', end_date=None):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.scaler = MinMaxScaler()
        self.data = None
        self.scaled_data = None

    def get_data(self):
        """Downloads data from Yahoo Finance"""

        print('Downloading data from Yahoo Finance for {}'.format(self.ticker))
        df = yf.download(tickers=self.ticker, start=self.start_date, end=self.end_date)
        self.data = df[['Close']].values
        print('Downloaded data: {} records'.format(len(self.data)))
        return self.data

    def normalize_data(self):
        """Normalizes data"""

        if self.data is None:
            print("First download the data")
        self.scaled_data = self.scaler.fit_transform(self.data)
        return self.scaled_data

    def create_sequences(self, sequence_length):
        """Creates sequences for RNN training
        X - sequence from sequence_length days
        y - the value for the next day
        """

        X, y = [], []
        data = self.scaled_data

        for i in range(sequence_length, len(data)):
            X.append(data[i-sequence_length:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def get_train_val_test_split(self, X, y, train_ratio=0.7, val_ratio=0.15):
        """
        Splits data into training, validation, and test sets chronologically.
        """
        X = np.array(X)
        y = np.array(y)

        total_len = len(X)

        train_end = int(total_len * train_ratio)
        val_end = int(total_len * (train_ratio + val_ratio))

        X_train, y_train = X[:train_end], y[:train_end]

        X_val, y_val = X[train_end:val_end], y[train_end:val_end]

        X_test, y_test = X[val_end:], y[val_end:]

        # Reshape for LSTM [samples, time steps, features]
        def reshape_data(data):
            return np.reshape(data, (data.shape[0], data.shape[1], 1))

        X_train = reshape_data(X_train)
        X_val = reshape_data(X_val)
        X_test = reshape_data(X_test)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)


# Class test
if __name__ == '__main__':

    processor = YahooDataGetter(ticker='AAPL')

    raw_data = processor.get_data()
    processor.normalize_data()

    SEQUENCE_LENGTH = 60
    X, y = processor.create_sequences(SEQUENCE_LENGTH)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.get_train_val_test_split(X, y)

    print("Data formats:")
    print(f"X_train shape: {X_train.shape} (Samples, Time steps, Features)")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape:  {X_test.shape}")
    print(f"y_test shape:  {y_test.shape}")
    print(f"X_val shape:   {X_val.shape}")
    print(f"y_val shape:   {y_val.shape}")