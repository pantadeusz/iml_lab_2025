import json
import requests
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error


SYMBOL = "AAPL"
INTERVAL = "1wk"   
LOOKBACK = 20        
TRAIN_SPLIT = 0.8


def load_data(symbol, interval):
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{symbol}?interval={interval}&range=5y&includeAdjustedClose=true"
    )

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
        "Accept": "application/json"
    }

    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
    
    data = response.json()
    return data


def extract_close_prices(json_data):
    result = json_data["chart"]["result"][0]
    closes = result["indicators"]["quote"][0]["close"]
    closes = [c for c in closes if c is not None]
    return np.array(closes).reshape(-1, 1)


def create_sequences(data, lookback):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback])
    return np.array(X), np.array(y)


def main():
    print("Pobieranie danych...")
    raw_data = load_data(SYMBOL, INTERVAL)
    prices = extract_close_prices(raw_data)

    scaler = MinMaxScaler()
    prices_scaled = scaler.fit_transform(prices)

    X, y = create_sequences(prices_scaled, LOOKBACK)

    split_idx = int(len(X) * TRAIN_SPLIT)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    model = Sequential([
        LSTM(50, activation="tanh", input_shape=(LOOKBACK, 1)),
        Dense(1)
    ])

    model.compile(
        optimizer="adam",
        loss="mse"
    )

    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    y_pred_test = model.predict(X_test)
    mse = mean_squared_error(
        scaler.inverse_transform(y_test),
        scaler.inverse_transform(y_pred_test)
    )

    print(f"\nsredni błąd kwadratowy mse na zbiorze testowym: {mse:.4f}")

    last_sequence = prices_scaled[-LOOKBACK:].reshape(1, LOOKBACK, 1)
    next_prediction_scaled = model.predict(last_sequence)
    next_prediction = scaler.inverse_transform(next_prediction_scaled)

    print(f"\nprzewidywana cena w następnym kroku czasowym: {next_prediction[0][0]:.2f}")

if __name__ == "__main__":
    main()
