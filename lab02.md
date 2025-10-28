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

## Podsumowanie

Po wykonaniu ćwiczeń proszę jak zwykle umieścić kod na repozytorium w branchu i zrobić PR. Każde zadanie to osobna para .py oraz .md.
