from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo

def load_and_split_data():
    wine = fetch_ucirepo(id=109)
    X = wine.data.features
    y = wine.data.targets
    print(type(X))
    print(y)
    return train_test_split(X, y, test_size=.2)

def get_rf_model():
    pass

if __name__ == '__main__':
    X_train, X_val, y_train, y_val = load_and_split_data()