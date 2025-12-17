import pandas as pd
import tensorflow as tf
import requests

from lab11_WindowGenerator import WindowGenerator

def get_data_from_yahoo(symbol, range, interval):
    """
        Getting JSON data from Yahoo
    """
    URL = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval={interval}&range={range}&symbol={symbol}"
    response = requests.get(URL, headers={'User-Agent': 'Mozilla/5.0'})

    if response.status_code == 200:
        data = response.json()
        result = data['chart']['result'][0]
        timestamps = result['timestamp']
        quote = result['indicators']['quote'][0]
        close_prices = quote['close']

        df = pd.DataFrame({'Price': close_prices})
        df.index = pd.to_datetime(timestamps, unit='s')
        return df.dropna()
    else:
        print(f"Error: {response.status_code}")

def test_train_val_split(dataframe, normalization=True):
    n = len(dataframe)
    train_df = dataframe[0:int(n*0.7)]
    val_df = dataframe[int(n*0.7):int(n*0.9)]
    test_df = dataframe[int(n*0.9):]

    if normalization:
        train_mean = train_df.mean()
        train_std = train_df.std()

        train_df = (train_df - train_mean) / train_std
        val_df = (val_df - train_mean) / train_std
        test_df = (test_df - train_mean) / train_std

        return test_df, train_df, val_df, train_mean, train_std

    return test_df, train_df, val_df

def create_model():
    model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    return model

def compile_and_fit(model, window, patience=2):
  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=patience,
                                                    mode='min')

  model.compile(loss=tf.keras.losses.MeanSquaredError(),
                optimizer=tf.keras.optimizers.Adam(),
                metrics=[tf.keras.metrics.MeanAbsoluteError()])

  history = model.fit(window.train, epochs=20,
                      validation_data=window.val,
                      callbacks=[early_stopping])
  return history

def main():
    data = get_data_from_yahoo('AAPL', "5y", "1d")
    test_df, train_df, val_df, train_mean, train_std = test_train_val_split(data, normalization=True)
    model = create_model()

    size_of_window = 24

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    data_window = WindowGenerator(input_width=size_of_window, label_width=1, shift=1,
                         train_df=train_df, val_df=val_df, test_df=test_df,
                         label_columns=['Price'])

    history = compile_and_fit(model, data_window)

    print("\n--- Wyniki na zbiorze testowym ---")
    val_performance = model.evaluate(data_window.val)
    test_performance = model.evaluate(data_window.test, verbose=0)
    print(f"Val: {val_performance}, \nTest: {test_performance}")
    print(f"MSE (znormalizowane): {test_performance[0]:.4f}")

    last_window_raw = data.iloc[-size_of_window:]
    last_window_norm = (last_window_raw - train_mean) / train_std

    input_tensor = tf.convert_to_tensor(last_window_norm.values, dtype=tf.float32)
    input_tensor = input_tensor[tf.newaxis, ...]

    prediction_norm = model.predict(input_tensor)

    prediction_real = (prediction_norm * train_std['Price']) + train_mean['Price']
    last_real_price = data.iloc[-1]['Price']

    print(f"Ostatnia cena (zamknięcie): {last_real_price:.2f}")
    print(f"Przewidywana cena (następna): {prediction_real[0][0]:.2f}")


if __name__ == "__main__":
    main()