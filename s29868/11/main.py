import tensorflow as tf
import numpy as np
from keras.src.layers import Dropout
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM, Dense

from yahooDataGetter import YahooDataGetter


def create_model(X_train):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),

        LSTM(units=50, return_sequences=False),
        Dropout(0.2),

        Dense(units=25),
        Dense(units=1)
    ])
    return model


def compile_fit_model(model, X_train, y_train, X_val, y_val, patience=5):
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )

    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping],
    )
    return history


def evaluate_and_predict(model, X_test, y_test, getter):
    test_loss = model.evaluate(X_test, y_test, verbose=0)
    print(f"Mean Squared Error on Test Set: {test_loss[0]:.6f}")

    last = getter.scaled_data[-60:]
    last_reshaped = np.reshape(last, (1, 60, 1))

    last_real_price = getter.data[-1][0]
    predicted_scaled = model.predict(last_reshaped)
    predicted_price = getter.scaler.inverse_transform(predicted_scaled)
    print(f"Last Real Price: {last_real_price:.6f}")
    print(f"Predicted Price: {predicted_price}")

    return predicted_price[0][0]

if __name__ == "__main__":
    getter = YahooDataGetter(ticker="AAPL")
    getter.get_data()
    getter.normalize_data()

    X, y = getter.create_sequences(sequence_length=60)

    (X_train, y_train), (X_val, y_val), (X_test, y_test) = getter.get_train_val_test_split(X, y)

    model = create_model(X_train)
    history = compile_fit_model(model, X_train, y_train, X_val, y_val)
    evaluate_and_predict(model, X_test, y_test, getter)


