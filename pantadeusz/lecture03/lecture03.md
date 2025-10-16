# Do zrobienia w ramach wykładu

Na wykładzie chciałbym wykonać wstępnie takie ćwiczenie wraz z omówieniem.

## Ćwiczenie 2: Krzywe ROC i Precision-Recall

**Cel**

Wygenerować i porównać krzywe ROC oraz Precision-Recall dla różnych modeli. Obliczyć AUC.

**Kroki**

1. Wybierz zbiór danych do nauki.
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
