import json
import os
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import keras
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import argparse


def load_json_data(file_path):
    """Load JSON data from file."""
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{file_path}'.")
        sys.exit(1)


def extract_time_series_data(data):
    """Extract time series data from Yahoo Finance JSON structure."""
    try:
        # Extract the relevant parts from the Yahoo Finance data structure
        result = data['chart']['result'][0]
        meta = result['meta']
        timestamps = result['timestamp']
        quotes = result['indicators']['quote'][0]
        
        close_prices = quotes['close']
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'close_price': close_prices
        })
        
        # Convert timestamps to dates
        df['date'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Sort by date
        df = df.sort_values('date').reset_index(drop=True)
        
        # Remove any NaN values
        df = df.dropna()
        
        return df
    except KeyError as e:
        print(f"Error: Missing key in JSON data: {e}")
        sys.exit(1)


def prepare_training_data(df, sequence_length=60):
    """Prepare data for training the model."""
    # Use closing prices for prediction
    prices = df['close_price'].values.reshape(-1, 1)
    
    # Check if we have enough data
    if len(prices) <= sequence_length:
        raise ValueError(f"Not enough data points ({len(prices)}) for sequence length ({sequence_length})")
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_prices = scaler.fit_transform(prices)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(scaled_prices)):
        X.append(scaled_prices[i-sequence_length:i, 0])
        y.append(scaled_prices[i, 0])
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Handle empty arrays
    if len(X) == 0:
        raise ValueError("Not enough data to create sequences. Try reducing sequence_length.")
    
    # Reshape for LSTM input (samples, time steps, features)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler


def build_lstm_model(input_shape):
    """Build LSTM model for time series forecasting."""
    model = keras.Sequential([
        keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(units=50, return_sequences=True),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(units=50, return_sequences=False),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(units=25),
        keras.layers.Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_model(X_train, y_train, epochs=50, batch_size=32):
    """Train the LSTM model."""
    model = build_lstm_model((X_train.shape[1], 1))
    
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    return model


def save_model(model, scaler, model_path, scaler_path):
    """Save trained model and scaler."""
    model.save(model_path)
    with open(scaler_path, 'wb') as f:
        import pickle
        pickle.dump(scaler, f)


def load_model_and_scaler(model_path, scaler_path):
    """Load trained model and scaler."""
    model = keras.models.load_model(model_path)
    with open(scaler_path, 'rb') as f:
        import pickle
        scaler = pickle.load(f)
    return model, scaler


def predict_future_prices(model, scaler, df, days_ahead=1, sequence_length=60):
    """Predict future prices."""
    # Get the last sequence of data
    prices = df['close_price'].values.reshape(-1, 1)
    scaled_prices = scaler.transform(prices)
    
    # Get the last sequence
    last_sequence = scaled_prices[-sequence_length:].reshape(1, sequence_length, 1)
    
    # Make predictions
    predictions = []
    current_sequence = last_sequence.copy()
    
    for _ in range(days_ahead):
        # Predict next price
        pred = model.predict(current_sequence, verbose=0)
        predictions.append(pred[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.roll(current_sequence, -1)
        current_sequence[0, -1, 0] = pred[0, 0]
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions = scaler.inverse_transform(predictions)
    
    return predictions.flatten()


def main():
    """Main function to run the application."""
    parser = argparse.ArgumentParser(description="Yahoo Finance Time Series Forecasting App")
    parser.add_argument("--input-file", "-i", required=True, help="Path to JSON file with Yahoo Finance data")
    parser.add_argument("--model-path", "-m", default="model.keras", help="Path to save/load model")
    parser.add_argument("--scaler-path", "-s", default="scaler.pkl", help="Path to save/load scaler")
    parser.add_argument("--train", "-t", action="store_true", help="Train the model")
    parser.add_argument("--predict", "-p", type=int, default=1, help="Number of days to predict")
    parser.add_argument("--sequence-length", "-l", type=int, default=60, help="Sequence length for LSTM")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of training epochs")
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    data = load_json_data(args.input_file)
    
    # Extract time series data
    print("Extracting time series data...")
    df = extract_time_series_data(data)
    print(f"Loaded {len(df)} data points")
    
    # Check if model exists or training is requested
    # Rewrite the deffault for defaults with more meaning based on the input file name
    if args.model_path == "model.keras":
        args.model_path = f"{Path(args.input_file).stem}.keras"

    if args.scaler_path == "scaler.pkl":
        args.scaler_path = f"{Path(args.input_file).stem}.pkl"

    model_exists = os.path.exists(args.model_path) and os.path.exists(args.scaler_path)
    
    if args.train or not model_exists:
        print("Training model...")
        try:
            X, y, scaler = prepare_training_data(df, args.sequence_length)
            
            # Split data (use last 20% for validation)
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            model = train_model(X_train, y_train, args.epochs)
            
            # Save model and scaler
            save_model(model, scaler, args.model_path, args.scaler_path)
            print(f"Model saved to {args.model_path}")
            print(f"Scaler saved to {args.scaler_path}")
            
            # Evaluate on validation set
            val_loss = model.evaluate(X_val, y_val, verbose=0)
            print(f"Validation loss: {val_loss}")
        except ValueError as e:
            print(f"Error during training: {e}")
            sys.exit(1)
    else:
        print("Loading existing model...")
        model, scaler = load_model_and_scaler(args.model_path, args.scaler_path)
    
    # Make predictions if requested
    if args.predict > 0:
        print(f"Making predictions for {args.predict} days ahead...")
        predictions = predict_future_prices(model, scaler, df, args.predict, args.sequence_length)
        
        # Display predictions
        print("\nPredictions:")
        for i, pred in enumerate(predictions):
            future_date = datetime.now().replace(microsecond=0) + pd.Timedelta(days=i+1)
            print(f"{future_date.strftime('%Y-%m-%d')}: ${pred:.2f}")


if __name__ == "__main__":
    main()
