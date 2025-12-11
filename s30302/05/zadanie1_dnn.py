import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from keras_tuner import RandomSearch

def load_data():
    data = load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = MinMaxScaler()
    X_train_scalered = scaler.fit_transform(X_train)
    X_test_scalered = scaler.transform(X_test)

    return X_train_scalered, X_test_scalered, y_train, y_test

def train_baseline(X_train, y_train):
    model_baseline = RandomForestClassifier(random_state=42)
    model_baseline.fit(X_train, y_train)

    return model_baseline

def build_model(hp):
    model = tf.keras.Sequential()

    for i in range(hp.Int('num_layers', 1, 3)):
        model.add(Dense(
            units=hp.Int('units', min_value=32, max_value=512, step=32),
            activation=hp.Choice('activation', ['relu', 'tanh'])
        ))
    model.add(Dense(3, activation='softmax'))

    optimizer_choice = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='log')

    optimizer_class = {
        'adam': tf.keras.optimizers.Adam,
        'rmsprop': tf.keras.optimizers.RMSprop,
        'sgd': tf.keras.optimizers.SGD
    }[optimizer_choice]

    optimizer = optimizer_class(learning_rate=learning_rate)


    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_dnn(X_train, y_train, X_val, y_val):
    tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=15,
        executions_per_trial=1,
        directory='tuner_dir',
        project_name='iris_dnn_tuning',
    )

    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=16,
        verbose=1,
    )

    best_model = tuner.get_best_models(num_models=1)[0]

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("Najlepsze hiperparametry znalezione przez tuner:")
    print(f"num_layers: {best_hps.get('num_layers')}")
    print(f"units: {best_hps.get('units')}")
    print(f"activation: {best_hps.get('activation')}")
    print(f"learning_rate: {best_hps.get('learning_rate')}")

    best_model.save("best_dnn_model.keras")
    print("Zapisano najlepszy model do pliku: best_dnn_model.keras")

    return best_model

def evaluate_model(model, X_test, y_test):
    model_name = type(model).__name__
    print(f"Evaluation for {model_name}:")

    y_pred = model.predict(X_test)

    if isinstance(y_pred, np.ndarray) and y_pred.ndim == 2:
        y_pred = y_pred.argmax(axis=1)

    acc = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return acc

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    model_baseline = train_baseline(X_train, y_train)

    model_dnn = train_dnn(X_train, y_train, X_test, y_test)

    acc_baseline = evaluate_model(model_baseline, X_test, y_test)
    print(f"Baseline acc: {acc_baseline}")

    acc_dnn = evaluate_model(model_dnn, X_test, y_test)
    print(f"DNN acc: {acc_dnn}")