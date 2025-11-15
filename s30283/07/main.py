from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import joblib
import os

def load_and_split_data():
    wine = fetch_ucirepo(id=109)
    X = wine.data.features
    y = wine.data.targets
    X = X.values
    y = np.squeeze(y.values)
    return train_test_split(X, y, test_size=.2)

def get_rf_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=20)
    model.fit(X_train, y_train)
    return model

def create_and_evaluate_rf_model(X_train, X_val, y_train, y_val):
    model = get_rf_model(X_train, y_train)
    y_preds = model.predict(X_val)
    print(f'RandomForest val accuarcy: {accuracy_score(y_val, y_preds)}')
    joblib.dump(model, 'rf_model.joblib')
    print(f'RandomForest model size: {os.path.getsize('rf_model.joblib') / 1024} bytes')

def create_and_evaluate_nn_model(X_train, X_val, y_train, y_val, first_layer_units, second_layer_units, epochs):
    y_train = y_train - 1
    y_val = y_val - 1
    normalization_layer = tf.keras.layers.Normalization()
    normalization_layer.adapt(X_train)
    model = tf.keras.Sequential([
        normalization_layer,
        tf.keras.layers.Dense(first_layer_units, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(second_layer_units, activation='relu'),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val))
    probs = model.predict(X_val)
    y_pred = np.argmax(probs, axis=1)
    print(accuracy_score(y_val, y_pred))
    model_filename = f'{first_layer_units}_{second_layer_units}_norm.keras'
    model.save(model_filename)
    print(f'Model size: {os.path.getsize(model_filename) / 1024} bytes')
    return history

def plot_acc_history(history):
    plt.figure(figsize=(10, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='magenta')
    plt.title('Model Accuracy Over Epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    X_train, X_val, y_train, y_val = load_and_split_data()
    create_and_evaluate_rf_model(X_train, X_val, y_train, y_val)
    history = create_and_evaluate_nn_model(X_train, X_val, y_train, y_val, 20, 6, epochs=100)
    plot_acc_history(history)