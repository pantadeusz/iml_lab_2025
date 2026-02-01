from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
import joblib
import os

SEED = 42

def prepare_data():
    wine = fetch_ucirepo(id=109)

    X = wine.data.features.values
    y = wine.data.targets.values.ravel() - 1

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_test = scaler.transform(X_test)

    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=0.1, random_state=SEED, stratify=y_tr
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_rf_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=SEED)
    model.fit(X_train, y_train)
    val_accuracy = model.score(X_val, y_val)
    test_accuracy = model.score(X_test, y_test)
    y_pred = model.predict(X_test)

    print("\n=== Random Forest ===")
    print(f"\nValidation Accuracy: {val_accuracy:.4f}")
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    model_filename = "wine_random_forest.pkl"
    joblib.dump(model, model_filename)

    model_size = os.path.getsize(model_filename)
    print(f"  Rozmiar: {model_size:,} bajtów ({model_size / 1024:.2f} KB / {model_size / (1024 * 1024):.2f} MB)")

    return model



def main():
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_data()
    rf_model = build_rf_and_evaluate(X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    main()