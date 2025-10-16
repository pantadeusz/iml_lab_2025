# Ćwiczenia do Wykładu 2: Wiedza Niedoskonała w Uczeniu Maszynowym

## Wprowadzenie

Te ćwiczenia są zaprojektowane, aby pomóc studentom zrozumieć i zastosować koncepcje z wykładu 2 dotyczące wiedzy niedoskonałej w uczeniu maszynowym. Ćwiczenia skupiają się na praktycznym zastosowaniu bibliotek scikit-learn i matplotlib do analizy danych, oceny modeli i radzenia sobie z niedoskonałościami takimi jak braki danych, szum, niezbalansowane klasy itp.

### Wymagania

- **Środowisko**: Lokalne środowisko Python (VS Code lub PyCharm).
- **Biblioteki**: scikit-learn, matplotlib, numpy, pandas (opcjonalnie dla zewnętrznych danych).
- **Dane**: Użyj zbiorów danych z scikit-learn (np. `load_breast_cancer`, `make_classification`) lub zewnętrznych źródeł (np. UCI Machine Learning Repository, Kaggle). Pobierz dane lokalnie i załaduj je za pomocą pandas.
- **Clean Code**: Przestrzegaj zasad czystego kodu (czytelne nazwy zmiennych, funkcje, komentarze). Użyj formattera jak `black` do formatowania kodu.

## Ćwiczenie 1: Macierz Pomyłek i Miary Klasyfikacji

**Cel**

Zaimplementować obliczenie macierzy pomyłek i miar klasyfikacji binarnej (czułość, swoistość, precyzja, NPV, dokładność, F1) ręcznie i za pomocą scikit-learn. Porównać wyniki.

**Kroki**

1. Załaduj zbiór danych binarnej klasyfikacji (np. `load_breast_cancer` z scikit-learn).
2. Wytrenuj klasyfikator (np. LogisticRegression).
3. Oblicz predykcje na zbiorze testowym.
4. Zaimplementuj funkcję do ręcznego obliczenia macierzy pomyłek i miar.
5. Użyj `confusion_matrix` i `classification_report` z scikit-learn do weryfikacji.
6. Wizualizuj macierz pomyłek za pomocą `ConfusionMatrixDisplay`.

**Kod struktura**

```python
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

# Załaduj dane
data = load_breast_cancer()
X, y = data.data, data.target
# Podziel dane
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Trenuj model
model = LogisticRegression(random_state=42, max_iter=3000)
model.fit(X_train, y_train)
# Predykcje
y_pred = model.predict(X_test)


# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    # Implementacja macierzy pomyłek i miar
    pass


def manual_classification_report(y_true, y_pred, output_dict=True):
    confusion_matrix = manual_confusion_matrix(y_true, y_pred)
    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report
    pass


# Użyj scikit-learn
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm2 = manual_confusion_matrix(y_test, y_pred)
print(cm2)
print(classification_report(y_test, y_pred, output_dict=True))
print(manual_classification_report(y_test, y_pred, output_dict=True))
# Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png")
plt.close()
```

**Eksperymenty**

- Zastosuj `predict_proba` i ustaw pozwól na ustawienie progu decyzji - powinno to być jako argument linii komend.
- Obserwuj wpływ progu na czułość i swoistość.
- Napisz małe podsumowanie w pliku .md

## Ćwiczenie 2: Krzywe ROC i Precision-Recall

**Cel**

Wygenerować i porównać krzywe ROC oraz Precision-Recall dla różnych modeli. Obliczyć AUC.

**Kroki**

1. Użyj tego samego zbioru danych co w ćwiczeniu 1.
2. Wytrenuj kilka modeli (LogisticRegression, RandomForest, SVC z probability=True).
3. Dla każdego modelu oblicz `roc_curve`, `auc`, `precision_recall_curve`.
4. Wizualizuj krzywe na jednym wykresie.
5. Porównaj AUC dla ROC i PR.

**Kod struktura**

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve

models = [
    ('Logistic Regression', LogisticRegression(random_state=42)),
    ('Random Forest', RandomForestClassifier(random_state=42)),
    ('SVM', SVC(probability=True, random_state=42))
]

plt.figure(figsize=(12, 5))

for name, model in models:
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
    
    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')

plt.subplot(1, 2, 1)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()

plt.show()
```

**Eksperymenty**

- Zastosuj na niezbalansowanym zbiorze (np. użyj `make_classification` z `weights=[0.9, 0.1]`).
- Eksperymentuj z parametrami modeli (np. `max_depth` dla RandomForest).
- Obserwuj różnice między ROC a PR AUC przy niezbalansowaniu.
- Napisz małe podsumowanie w pliku .md

## Ćwiczenie 3: Radzenie Sobie z Brakującymi Danymi

**Cel**

Zastosować różne metody imputacji braków danych (MCAR/MAR/MNAR symulacja) i ocenić wpływ na model.

**Kroki**

1. Wybierz zbiór danych (np. `load_diabetes` lub zewnętrzny z brakami).
2. Symuluj braki danych (np. MCAR: losowo usuń wartości).
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
- I znowu - małe podsumowanie w .md

## Ćwiczenie 4: Niezbalansowane Klasy

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


## Ćwiczenie 5: Przykład Testu Medycznego (Bayes w Praktyce)

**Cel**

Obliczyć prawdopodobieństwa z przykładu testu medycznego z wykładu i wizualizować wpływ parametrów.

**Kroki**

1. Zaimplementuj funkcję do obliczania p(Choroba|Test+) używając Twierdzenia Bayesa.
2. Eksperymentuj z różnymi wartościami czułości, swoistości, częstości choroby.
3. Wizualizuj jak zmienia się posterior w zależności od parametrów.

**Kod struktura**

```python
def bayes_medical_test(prior, sensitivity, specificity):
    # Oblicz p(Choroba|Test+)
    p_test_given_disease = sensitivity
    p_test_given_no_disease = 1 - specificity
    p_disease = prior
    p_no_disease = 1 - prior
    
    numerator = p_test_given_disease * p_disease
    denominator = numerator + p_test_given_no_disease * p_no_disease
    posterior = numerator / denominator
    return posterior

# Przykład z wykładu
post = bayes_medical_test(0.01, 0.99, 0.95)
print(f'P(Choroba|Test+) = {post:.3f}')
```

**Eksperymenty**

- Zmień częstość choroby (np. 0.001, 0.1).
- Obserwuj wpływ fałszywie dodatnich.
- Rozszerz na więcej testów w sekwencji.

## Podsumowanie

Po wykonaniu ćwiczeń proszę jak zwykle umieścić kod na repozytorium w branchu i zrobić PR. Każde zadanie to osobna para .py oraz .md.
