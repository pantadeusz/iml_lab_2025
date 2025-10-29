import kagglehub
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import seaborn as sns


def download_dataset():
    path = kagglehub.dataset_download("ahmeduzaki/earthquake-alert-prediction-dataset")
    path = os.path.join(path, 'earthquake_alert_balanced_dataset.csv')
    print("🚨 Dataset downloaded")
    return path

def load_data(path):
    data = pd.read_csv(path)
    print("🚨 Data loaded")
    return data

def preprocess_data(data):
    target = 'alert'
    X, y = data[[col for col in data if col != target]], data[target]
    
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("🚨 Data preprocessed")
    
    return X_train_scaled, X_test_scaled, y_train, y_test

def get_random_forest(X_train, y_train):
    random_forest_model = RandomForestClassifier(n_estimators=50,
                                                 criterion='log_loss')
    random_forest_model.fit(X_train, y_train)
    return random_forest_model

def prepare_loaders(X_train, X_test, y_train, y_test):
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = 32
    train_ds = (train_ds
                .shuffle(buffer_size=len(X_train))
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (test_ds
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))
    
    print("🚨 Loaders prepared")
    return train_ds, test_ds

def get_neural_network(X_train, X_test, y_train, y_test):
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

    nn_model.fit(train_ds, validation_data=test_ds, epochs=500)
    return nn_model, test_ds

def compare_models(rf_model, nn_model, X_test, y_test, test_ds):
    rf_preds = rf_model.predict(X_test)
    nn_preds = np.argmax(nn_model.predict(test_ds), axis=1)
    
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(y_test, rf_preds), 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Random Forest Confusion Matrix')
    
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(y_test, nn_preds), 
                annot=True, fmt='d', cmap='Reds')
    plt.title('Neural Network Confusion Matrix')
    
    print("\n=== Random Forest Performance ===")
    print(classification_report(y_test, rf_preds))
    
    print("\n=== Neural Network Performance ===")
    print(classification_report(y_test, nn_preds))
    
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("🚨 Starting process")
    
    path = download_dataset()
    data = load_data(path)
    
    X_train, X_test, y_train, y_test = preprocess_data(data)
    print(f'X_train shape: {X_train.shape}\ny_train shape: {y_train.shape}')
    
    rf_model = get_random_forest(X_train, y_train)
    nn_model, test_ds = get_neural_network(X_train, X_test, y_train, y_test)
    
    compare_models(rf_model, nn_model, X_test, y_test, test_ds)
    print("🚨 Models compared")
    
    print("🚨 Process finished")
