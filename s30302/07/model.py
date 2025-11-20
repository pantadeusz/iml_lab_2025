from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle
import os
from tensorflow import keras
from tensorflow.keras import layers

def load_data():
    wine = fetch_ucirepo(id=109)

    X = wine.data.features
    y = wine.data.targets

    y = y - 1

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    num_classes = y.nunique()

    return X_train, X_test, y_train, y_test, num_classes

def random_forest_classifier_model_init():
    model = RandomForestClassifier(random_state=42)

    return model

def random_forest_classifier_model_train(model, X_train, y_train):
    model.fit(X_train, y_train)

def random_forest_classifier_model_predict(model, X_test):
    y_pred = model.predict(X_test)

    return y_pred

def random_forest_classifier_model_evaluation(y_test, y_pred):
    print(classification_report(y_test, y_pred))

def random_forest_classifier_model_save(model):
    with open("model.pkl", "wb") as f:
        pickle.dump(model, f)

    size = os.path.getsize("model.pkl")
    print(f"Rozmiar modelu: {size / 1024:.2f} KB")

def build_three_layer_model(num_classes, normalizer):
    model = keras.Sequential(
        [
            normalizer,
            layers.Dense(16, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),

    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def three_layer_model_train(model, X_train, y_train, X_test, y_test):
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=20,
        batch_size=16,
        verbose=1
    )

def three_layer_model_predict_and_evaluation(model, X_test, y_test):
    loss, accuracy = model.evaluate(X_test, y_test)

    return loss, accuracy

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, num_classes = load_data()
    X_train_np = X_train.to_numpy()
    X_test_np = X_test.to_numpy()
    y_train_np = y_train.to_numpy()
    y_test_np = y_test.to_numpy()

    num_classes = int(num_classes)

    normalizer = layers.Normalization()
    normalizer.adapt(X_train_np)

    three_layer_model = build_three_layer_model(num_classes, normalizer)
    three_layer_model_train(three_layer_model, X_train_np, y_train_np, X_test_np, y_test_np)
    loss, accuracy = three_layer_model_predict_and_evaluation(three_layer_model, X_test_np, y_test_np)

    print("after evaluation metrics:")
    print("Accuracy:", accuracy)
    print("Loss:", loss)

