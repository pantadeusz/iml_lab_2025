from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from joblib import dump


# Rozmiar klasyfikatora = 212KB

def prepare_dataset():
    wine = fetch_ucirepo(id=109)

    X = wine.data.features
    y = wine.data.targets

    X_train, X_test, y_train, y_test = (train_test_split(X, y, test_size=0.2, random_state=42))

    y_train = y_train - 1
    y_test = y_test - 1

    return X_train, X_test, y_train, y_test

def learn(X_train, y_train, X_test):
    classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    classifier.fit(X_train, y_train.iloc[:, 0])
    y_pred = classifier.predict(X_test)

    # Save model
    dump(classifier, "random_forest.joblib")

    return y_pred


def main():
    X_train, X_test, y_train, y_test = prepare_dataset()

    y_pred = learn(X_train, y_train, X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {accuracy * 100:.2f}%')


if __name__ == '__main__':
    main()
