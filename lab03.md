# Ćwiczenia 3

## 1: Radzenie Sobie z Brakującymi Danymi

**Cel**

Zastosować różne metody imputacji braków danych (MCAR/MAR/MNAR symulacja) i ocenić wpływ na model.

**Kroki**

1. Wybierz zbiór danych (np. `load_diabetes` lub zewnętrzny z brakami).
2. Symuluj braki danych (sugeruję MCAR: losowo usuń wartości).
3. Zastosuj metody imputacji: średnia, KNN, MICE (IterativeImputer).
4. Wytrenuj model na danych z imputacją i bez.
5. Porównaj metryki.

**Kod struktura**

```python
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

# Załaduj dane
data = load_diabetes()
X, y = data.data, data.target

# Symuluj braki (MCAR)
rng = np.random.RandomState(42)
missing_mask = rng.rand(*X.shape) < 0.1  # 10% braków
X_missing = X.copy()
X_missing[missing_mask] = np.nan

# Metody imputacji
imputers = {
    'Mean': SimpleImputer(strategy='mean'),
    'KNN': KNNImputer(n_neighbors=5),
    'MICE': IterativeImputer(random_state=42)
}

for name, imputer in imputers.items():
    X_imputed = imputer.fit_transform(X_missing)
    # Trenuj model i oceń
    model = LogisticRegression(random_state=42)
    # ... (podziel na train/test, trenuj, oceń)
```

**Eksperymenty**

- Zmień procent braków (np. 5%, 20%).
- Porównaj wpływ na różne typy modeli.
- Małe podsumowanie w .md

## 2: Niezbalansowane Klasy

**Cel**

Zastosować techniki radzenia sobie z niezbalansowanymi klasami i ocenić wpływ na metryki.

**Kroki**

1. Użyj `make_classification` z niezbalansowaniem (np. `weights=[0.95, 0.05]`).
2. Wytrenuj bazowy model i oceń metryki.
3. Zastosuj: ważenie klas, oversampling (SMOTE), undersampling.
4. Porównaj wyniki.

**Kod przykład/sugestia**

```python
from sklearn.datasets import make_classification

# Generuj dane niezbalansowane
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)

# Bazowy model
model = LogisticRegression(random_state=42)
# ... trenuj i oceń

# Z ważeniem klas
model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
# ... 

```

**Zadanie**

- Porównaj metryki dla nauczonego modelu.
- Zobacz, czy za pomocą progowania uda się uzyskać dobre wyniki.
- Oczywiście małe podsumowanie w .md.

## Podsumowanie

Po wykonaniu ćwiczeń proszę jak zwykle umieścić kod na repozytorium w branchu i zrobić PR. Każde zadanie to osobna para .py oraz .md.
