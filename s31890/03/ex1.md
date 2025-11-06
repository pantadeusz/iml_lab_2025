## Dla zbalansowanych danych o następujących argumentach:
```
make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=2,
    n_redundant=10,
    n_clusters_per_class=1,
    flip_y=0,
    random_state=1
)
```
### Wynik:
ROC:
Logistic Regression: AUC = 0.95
Random Forest: AUC = 0.97
SVM: AUC = 0.97

Precision-Recall:
Logistic Regression: AUC = 0.95
Random Forest: AUC = 0.97
SVM: AUC = 0.96

## Dla niezbalansowanych danych o następujących argumentach:
```
make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=2,
    n_redundant=10,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    flip_y=0,
    random_state=1
)
```
### Wynik:
ROC:
Logistic Regression: AUC = 0.97
Random Forest: AUC = 0.96
SVM: AUC = 0.94

Precision-Recall:
Logistic Regression: AUC = 0.89
Random Forest: AUC = 0.91
SVM: AUC = 0.90

### Wniąski:
Przy zbalansowanym zbiorze danych wyniki AUC nie różniły się mocno pomiędzy ROC a PR.
Za to dla zbioru danych niezbalansowanych wyniki AUC różniły się pomiędzy ewaluacją ROC a PR. Wyniki ROC nie różniły się mocno pomiędzy zbalansowanym a nie zbalansowanym zbiorem danych.

Udowadnia to, że ewaluacja metodą ROC jest odpowiednia tylko gdy dane są zbalansowane, natomiast gdy nie są należy zastosować PR.
