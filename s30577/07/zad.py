from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
import joblib, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def load_data():
    wine = load_wine()
    X = wine.data.astype(np.float32)
    y = wine.target
    class_names = wine.target_names
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    return X_train, X_val, y_train, y_val

def RandomForest():
    X_train, X_val, y_train, y_val = load_data()

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    joblib.dump(rf, 'rf_wine')

def NeuralNetwork():
    X_train, X_val, y_train, y_val = load_data()

    model = keras.Sequential([
        layers.Input(shape=(13,)),
        layers.Dense(32, activation="relu"),
        layers.Dense(16, activation="relu"),
        layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=0)

    model.save('nn_basic.keras')

def NormalizedNeuralNetwork(u1 = 64, u2 = 32, epochs=50):

    X_train, X_val, y_train, y_val = load_data()

    normalizer = layers.Normalization()
    normalizer.adapt(X_train)

    model = keras.Sequential([
        normalizer,
        layers.Dense(u1, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.2),
        layers.Dense(u2, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(3, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=8,
        verbose=1)

    model.save(f'nn_normalized_{u1}_{u2}_e{epochs}.keras')


def CheckModelAccuracy(model):

    X_train, X_val, y_train, y_val = load_data()

    if model == 'RandomForest':
        rf = joblib.load('rf_wine')
        rf_size = os.path.getsize('rf_wine')
        print(f"Size of RF model: {rf_size} bytes")
        print(f"RandomForest Accuracy: {accuracy_score(y_val, rf.predict(X_val)):.4f}\n")

    else:
        model_path = f'{model}.keras'
        loaded_model = keras.models.load_model(model_path)
        model_size = os.path.getsize(model_path) 
        print(f"Size of {model_path}: {model_size} bytes")
        print(f"{model_path} Accuracy: {accuracy_score(y_val, loaded_model.predict(X_val, verbose=0).argmax(axis=-1)):.4f}\n")

def main():

    X_train, X_val, y_train, y_val = load_data()

    # RandomForest()
    #
    # NeuralNetwork()
    #
    # NormalizedNeuralNetwork(16,8,80)
    # NormalizedNeuralNetwork(16,8,50)
    # NormalizedNeuralNetwork(32,16,100)
    # NormalizedNeuralNetwork(32,16,200)
    # NormalizedNeuralNetwork(64,32,50)
    # NormalizedNeuralNetwork(32,16,100)
    # NormalizedNeuralNetwork(64,32,80)
    # NormalizedNeuralNetwork(128,64,50)
    # NormalizedNeuralNetwork(128,64,80)


    CheckModelAccuracy('RandomForest')
    CheckModelAccuracy('nn_basic')
    CheckModelAccuracy('nn_normalized_16_8_e50')
    # CheckModelAccuracy('nn_normalized_32_16_e100')
    # CheckModelAccuracy('nn_normalized_32_16_e200')
    # CheckModelAccuracy('nn_normalized_64_32_e50')
    # CheckModelAccuracy('nn_normalized_64_32_e80')
    # CheckModelAccuracy('nn_normalized_128_64_e50')
    # CheckModelAccuracy('nn_normalized_128_64_e80')

if __name__ == "__main__":
    main()