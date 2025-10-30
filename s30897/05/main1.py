import numpy as np
from keras.src.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


def load_data():
    diabetes = load_diabetes()
    X = diabetes.data
    Y = diabetes.target
    n_features = X.shape[1]

    median_val = np.median(Y)
    Y = np.where(Y >= median_val, 1, 0)

    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.3, random_state=300)

    scalar = StandardScaler()
    X_train_scaled = scalar.fit_transform(X_train)
    X_test_scaled = scalar.transform(X_test)

    print("-" * 30)
    print(f"Liczba danych w zbiorze treningowym: {X_train_scaled.shape}")
    print(f"Liczba danych w zbiorze testowym: {X_test_scaled.shape}")
    print("-" * 30)

    return X_train_scaled, X_test_scaled, y_train, y_test, n_features

def Random_Forest(X_train, X_test, y_train, y_test):
    rf_reg = RandomForestClassifier(n_estimators=100, random_state=100)
    rf_reg.fit(X_train, y_train)
    y_pred = rf_reg.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {"accuracy": accuracy,"confusion_matrix": cm, "classification_report": report}
def DNN(X_train, X_test, y_train, y_test, n_features):
    DNN_Model = Sequential()
    DNN_Model.add(Input(shape=(n_features,)))
    DNN_Model.add(Dense(units=64, activation='relu'))
    DNN_Model.add(Dropout(0.3))
    DNN_Model.add(Dense(units=32, activation='relu'))
    DNN_Model.add(Dropout(0.3))
    DNN_Model.add(Dense(units=16, activation='relu'))
    DNN_Model.add(Dense(units=1, activation='sigmoid'))

    DNN_Model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    DNN_Model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=0)

    y_pred = DNN_Model.predict(X_test)
    y_pred = (y_pred > 0.5).astype(int).flatten()

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    return {"accuracy": accuracy, "confusion_matrix": cm, "classification_report": report}
def compare_models(rf_metrics, dnn_metrics):
    print("-" * 30)
    print(f"Random forest - Accuracy: {rf_metrics['accuracy']:.4f}")
    print("\nMacierz Pomyłek:")
    print(rf_metrics["confusion_matrix"])
    print("\nRaport Klasyfikacji:")
    print(rf_metrics["classification_report"])
    print("-" * 30)

    print("-"*30)
    print(f"Model DNN - Accuracy: {dnn_metrics['accuracy']:.4f}")
    print("\nMacierz Pomyłek:")
    print(dnn_metrics["confusion_matrix"])
    print("\nRaport Klasyfikacji:")
    print(dnn_metrics["classification_report"])
    print("-"*30)

    if rf_metrics['accuracy'] > dnn_metrics['accuracy']:
        print(f"Wniosek: Random Forest poradził sobie lepiej na tym zbiorze danych. (accuracy: {rf_metrics['accuracy']:.4f} vs {dnn_metrics['accuracy']:.4f})")
    elif dnn_metrics['accuracy'] > rf_metrics['accuracy']:
        print(f"Wniosek: Sieć neuronowa (DNN) poradziła sobie lepiej. (accuracy: {rf_metrics['accuracy']:.4f} vs {dnn_metrics['accuracy']:.4f})")
    else:
        print(f"Wniosek: Oba modele osiągnęły bardzo podobny wynik. (accuracy: {rf_metrics['accuracy']:.4f} vs {dnn_metrics['accuracy']:.4f})")


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, n_features = load_data()

    rf_results = Random_Forest(X_train, X_test, y_train, y_test)
    dnn_results = DNN(X_train, X_test, y_train, y_test, n_features)
    compare_models(rf_results, dnn_results)

