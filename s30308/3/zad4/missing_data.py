from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import numpy as np

threshold = 0.2

def prepare_data(threshold_gap):
    # Załaduj dane
    data = load_diabetes()

    # X - są znormalizowane, y -są w wersji surowej
    X, y = data.data, data.target

    # Symuluj braki (MCAR)
    rng = np.random.RandomState(42)
    missing_mask = rng.rand(*X.shape) < threshold_gap  # x% braków
    X_missing = X.copy()
    X_missing[missing_mask] = np.nan

    # Normalizujemy y tak, aby były dwie klasy 0 i 1 (teraz jest wiele klas ponieważ jest wiele różnych wartości)
    threshold_gap = np.median(y)

    y = (y > threshold_gap).astype(int)

    return X, X_missing, y


def main():
    X_full, X_missing, y = prepare_data(threshold)

    # Wytrenuj model bez imputacji
    X_train, X_test, y_train, y_test = train_test_split(X_full, y, test_size=0.2, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc_full = roc_auc_score(y_test, y_prob)

    # Metody imputacji
    """
    Imputacja to proces uzupełniania brakujących wartości w danych. W wielu zbiorach danych spotyka się sytuacje, gdy 
    niektóre obserwacje są niepełne — np. ktoś nie wypełnił formularza, czujnik nie zmierzył wartości, 
    albo dane zostały utracone. Zamiast usuwać całe wiersze z brakami (co zmniejsza ilość danych), 
    stosuje się imputację, czyli “wstawianie” wartości zastępczych.
    """
    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'KNN': KNNImputer(n_neighbors=5),
        'MICE': IterativeImputer(random_state=42)
    }

    auc_scores = [auc_full]
    labels = ['Bez braków']

    for name, imputer in imputers.items():
        X_imputed = imputer.fit_transform(X_missing)
        # Trenuj model i oceń
        model = LogisticRegression(random_state=42)

        # ... (podziel na train/test, trenuj, oceń)
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)


        y_prob = model.predict_proba(X_test)[:, 1]

        # ROC AUC - pokazuje jakość predykcji probabilistycznej modelu
        auc = roc_auc_score(y_test, y_prob)

        auc_scores.append(auc)
        labels.append(name)

    plt.bar(labels, auc_scores, color='skyblue')
    plt.title(f"Porównanie AUC dla różnych metod imputacji (threshold={threshold})")
    plt.ylabel("AUC score")
    plt.show()

if __name__=="__main__":
    main()