import matplotlib.pyplot as plt
import tensorflow as tf
import yfinance as yf
import pandas as pd
import numpy as np
import argparse
import sys
import os
from time import time

class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                 train_df, val_df, test_df,
                 label_columns=None):
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def plot(self, model=None, plot_col='Close', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n+1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()

        plt.xlabel('Time [h]')
        filename = f'window_plot_{int(time())}.png'
        plt.savefig(filename)
        print(f"Saved plot: {filename}")
        plt.close()

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:
            result = next(iter(self.train))
            self._example = result
        return result

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)
        return ds

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


def init_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', required=True, type=str, help="Stock Ticker (e.g., INTC)")
    parser.add_argument('-s', '--steps', required=True, type=int, help="Number of steps (e.g., 20)")
    return parser

def download_data(stock_name):
    print(f"Downloading {stock_name}...")
    try:
        df = yf.download(tickers=stock_name, period='10y', interval='1d', auto_adjust=True)
    except Exception as e:
        print(f"Download Error: {e}")
        return pd.DataFrame()
    return df

def preprocess(df):
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel('Ticker')
    
    print('Dataset shape:', df.shape)
    
    if 'Close' not in df.columns and 'Close' in df.columns:
         pass
    
    data = df[['Close', 'Volume']].copy()
    
    date_time = pd.to_datetime(data.index)
    timestamp_s = date_time.map(pd.Timestamp.timestamp).to_numpy()
    
    # day = 24 * 60 * 60
    # year = (365.2425) * day
    # df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
    # df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
    # df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
    # df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))
    # plt.figure()
    # plt.plot(np.array(df['Day sin'])[:25])
    # plt.plot(np.array(df['Day cos'])[:25])
    # plt.savefig('year_signal.png')
    # plt.close()
    # plt.figure()
    # plt.plot(np.array(df['Year sin'])[:25])
    # plt.plot(np.array(df['Year cos'])[:25])
    # plt.savefig('year_signal.png')
    # plt.close()
    
    df['Volume'] = np.log1p(df['Volume'])
    
    plt.figure()
    plt.plot(df['Volume'])
    plt.title('Log Volume')
    plt.savefig('volume_log.png')
    plt.close()
    
    n = len(df)
    train_df = df[:int(n * 0.7)].copy()
    val_df = df[int(n * 0.7):int(n * 0.9)].copy()
    test_df = df[int(n * 0.9):].copy()
    
    plt.figure()
    plt.plot(data['Close'])
    plt.savefig('Close_graph.png')
    plt.close()
    
    train_mean = train_df.mean()
    train_std = train_df.std()

    train_df = (train_df - train_mean) / train_std
    val_df = (val_df - train_mean) / train_std
    test_df = (test_df - train_mean) / train_std
    
    return train_df, val_df, test_df

def create_last_baseline_model(steps):
    class MultiStepLastBaseline(tf.keras.Model):
        def call(self, inputs):
            return tf.tile(inputs[:, -1:, :], [1, steps, 1])

    last_baseline = MultiStepLastBaseline()
    last_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return last_baseline

def create_repeat_baseline_model(steps):
    class RepeatBaseline(tf.keras.Model):
        def call(self, inputs):
            return inputs
    repeat_baseline = RepeatBaseline()
    repeat_baseline.compile(loss=tf.keras.losses.MeanSquaredError(),
                        metrics=[tf.keras.metrics.MeanAbsoluteError()])
    return repeat_baseline

def create_linear_model(steps, num_features):
    linear_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(steps*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([steps, num_features])
    ])
    return linear_model

def create_dense_model(steps, num_features):
    dense_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(steps*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([steps, num_features])
    ])
    return dense_model

def create_conv_model(steps, num_features):
    CONV_WIDTH = 3
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
        tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
        tf.keras.layers.Dense(steps*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([steps, num_features])
    ])
    return conv_model

def create_lstm_model(steps, num_features):
    lstm_model = tf.keras.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=False),
        tf.keras.layers.Dense(steps*num_features,
                            kernel_initializer=tf.initializers.zeros()),
        tf.keras.layers.Reshape([steps, num_features])
    ])
    return lstm_model

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

def evaluate_models(steps, multi_window, num_features):
    val_performance = {}
    performance = {}
    
    print("\n--- Evaluating Last Baseline ---")
    last_baseline = create_last_baseline_model(steps)
    val_performance['Last'] = last_baseline.evaluate(multi_window.val, return_dict=True)
    performance['Last'] = last_baseline.evaluate(multi_window.test, verbose=0, return_dict=True)
    multi_window.plot(last_baseline)
    
    print("\n--- Evaluating Repeat Baseline ---")
    repeat_baseline = create_repeat_baseline_model(steps)
    val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val, return_dict=True)
    performance['Repeat'] = repeat_baseline.evaluate(multi_window.test, verbose=0, return_dict=True)
    multi_window.plot(repeat_baseline)
    
    print("\n--- Evaluating Linear Model ---")
    linear_model = create_linear_model(steps, num_features)
    history = compile_and_fit(linear_model, multi_window)
    val_performance['Linear'] = linear_model.evaluate(multi_window.val, return_dict=True)
    performance['Linear'] = linear_model.evaluate(multi_window.test, verbose=0, return_dict=True)
    multi_window.plot(linear_model)
    
    print("\n--- Evaluating Dense Model ---")
    dense_model = create_dense_model(steps, num_features)
    history = compile_and_fit(dense_model, multi_window)
    val_performance['Dense'] = dense_model.evaluate(multi_window.val, return_dict=True)
    performance['Dense'] = dense_model.evaluate(multi_window.test, verbose=0, return_dict=True)
    multi_window.plot(dense_model)
    
    print("\n--- Evaluating Conv Model ---")
    conv_model = create_conv_model(steps, num_features)
    history = compile_and_fit(conv_model, multi_window)
    val_performance['Conv'] = conv_model.evaluate(multi_window.val, return_dict=True)
    performance['Conv'] = conv_model.evaluate(multi_window.test, verbose=0, return_dict=True)
    multi_window.plot(conv_model)
    
    print("\n--- Evaluating LSTM Model ---")
    lstm_model = create_lstm_model(steps, num_features)
    history = compile_and_fit(lstm_model, multi_window)
    val_performance['LSTM'] = lstm_model.evaluate(multi_window.val, return_dict=True)
    performance['LSTM'] = lstm_model.evaluate(multi_window.test, verbose=0, return_dict=True)
    multi_window.plot(lstm_model)
    
    x = np.arange(len(performance))
    width = 0.3
    
    metric_name = 'mean_absolute_error'
    val_mae = [v[metric_name] for v in val_performance.values()]
    test_mae = [v[metric_name] for v in performance.values()]
    
    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.17, val_mae, width, label='Validation')
    plt.bar(x + 0.17, test_mae, width, label='Test')
    plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
    plt.ylabel('MAE (average over all times and outputs)')
    plt.legend()
    plt.savefig('results.png')
    plt.close()


def main():
    parser = init_argparser()
    args = parser.parse_args()
    stock_name, steps = args.name, args.steps
    
    df = download_data(stock_name)
    
    if df.empty:
        print("\n[!] FATAL ERROR: No data downloaded.")
        print("Please check your internet or update libraries: pip install --upgrade yfinance curl-cffi")
        sys.exit(1)
    
    train_df, val_df, test_df = preprocess(df)
    
    multi_window = WindowGenerator(input_width=steps,
                                   label_width=steps,
                                   shift=steps,
                                   train_df=train_df,
                                   val_df=val_df,
                                   test_df=test_df)

    multi_window.plot()
    print(multi_window)
    
    num_features = train_df.shape[1]
    evaluate_models(steps, multi_window, num_features)

if __name__ == '__main__':
    import yfinance
    try:
        import curl_cffi
        c_ver = curl_cffi.__version__
    except ImportError:
        c_ver = "Not Installed"

    print(f"Python Executable: {sys.executable}")
    print(f"Yfinance Version: {yfinance.__version__}")
    print(f"Curl_cffi Version: {c_ver}")
    print("-" * 30)
    
    main()