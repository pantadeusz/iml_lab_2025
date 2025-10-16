from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import r2_score
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

def prepare_data():
    # Załaduj dane
    data = load_diabetes()

    # X - są znormalizowane, y -są w wersji surowej
    X, y = data.data, data.target

    # Symuluj braki (MCAR)
    rng = np.random.RandomState(42)
    missing_mask = rng.rand(*X.shape) < 0.1  # 10% braków
    X_missing = X.copy()
    X_missing[missing_mask] = np.nan

    # Normalizujemy y tak, aby były dwie klasy 0 i 1 (teraz jest wiele klas ponieważ jest wiele różnych wartości)
    threshold = np.median(y)

    y = (y > threshold).astype(int)

    return X_missing, y


def main():
    X_missing, y = prepare_data()

    # Metody imputacji
    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'KNN': KNNImputer(n_neighbors=5),
        'MICE': IterativeImputer(random_state=42)
    }

    r2_scores = []

    for name, imputer in imputers.items():
        X_imputed = imputer.fit_transform(X_missing)
        # Trenuj model i oceń
        model = LogisticRegression(random_state=42)

        # ... (podziel na train/test, trenuj, oceń)
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=42)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # R² (współczynnik determinacji) mówi nam, jak dobrze model regresyjny dopasowuje się do danych.
        # Inaczej mówiąc: ile wariancji w danych model potrafi wyjaśnić. Posłuży do oceny imputera

        r2 = r2_score(y_test, y_pred)

        r2_scores.append(r2)

    labels = ['Mean', 'KNN', 'MICE']

    plt.bar(labels, r2_scores, color='skyblue')
    plt.title("Porównanie R² dla różnych metod imputacji")
    plt.ylabel("R² score")
    plt.show()

if __name__=="__main__":
    main()