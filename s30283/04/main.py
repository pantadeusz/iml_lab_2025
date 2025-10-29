import kagglehub
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt


def download_dataset():
    path = kagglehub.dataset_download("ahmeduzaki/earthquake-alert-prediction-dataset")
    path = os.path.join(path, 'earthquake_alert_balanced_dataset.csv')
    return path

def load_data(path):
    data = pd.read_csv(path)
    return data

def preprocess_data(data):
    target = 'alert'
    X, y = data[[col for col in data if col != target]], data[target]
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    
    return train_test_split(X, y_encoded, test_size=0.2)

def evaluate_random_forest(X_train, X_test, y_train, y_test):
    random_forest_model = RandomForestClassifier(n_estimators=50,
                                                 criterion='log_loss')
    random_forest_model.fit(X_train, y_train)
    preds = random_forest_model.predict(X_test)
    print(f'Random Forest acc: {(preds == y_test).mean().item()}')

def prepare_loaders(X_train, X_test, y_train, y_test):
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = 32
    train_ds = (train_ds
                .shuffle(buffer_size=len(X_train))
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (
        test_ds
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    exit()
    def normalize(x, y):
        return tf.cast(x, tf.float32) / 255.0, y  # example normalization

    train_ds = train_ds.map(normalize)
    test_ds = test_ds.map(normalize)
    return train_ds, test_ds

def make_plot(history):
    plt.plot(history.history['loss'])
    plt.show()

def evaluate_neural_network(X_train, X_test, y_train, y_test):
    train_ds, test_ds = prepare_loaders(X_train, X_test, y_train, y_test)

    nn_model = Sequential([
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(16, activation='relu'),
        Dense(4, activation='softmax')
    ])
    nn_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    history = nn_model.fit(train_ds, validation_data=test_ds, epochs=300)

    test_loss, test_acc = nn_model.evaluate(test_ds)
    print(f"Test accuracy: {test_acc:.3f}")

    y_pred_probs = nn_model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    print("Sample predictions:", y_pred[:10])

    make_plot(history)

if __name__ == '__main__':
    path = download_dataset()
    data = load_data(path)
    
    X_train, X_test, y_train, y_test = preprocess_data(data)
    print(f'X_train shape: {X_train.shape}\ny_train shape: {y_train.shape}')
    
    evaluate_random_forest(X_train, X_test, y_train, y_test)
    evaluate_neural_network(X_train, X_test, y_train, y_test)










