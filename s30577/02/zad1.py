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
import sys
import json

try:
    threshold = float(sys.argv[1])
except (IndexError, ValueError):
    threshold = 0.5

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

def manual_confusion_matrix(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    return np.array([[tn, fp], [fn, tp]])

def manual_classification_report(y_true, y_pred, output_dict=True):

    confusion_matrix = manual_confusion_matrix(y_true, y_pred)

    # na podstawowy punkt wystarczy: precision, recall, f1-score, support

    tn, fp, fn, tp = confusion_matrix.ravel()

    def dzielenieBezZera(licznik, mianownik):
        return licznik / mianownik if mianownik else 0.0

    accuracy = dzielenieBezZera(tp + tn, tp + tn + fp + fn)

    # dla klasy pozytywnej (1)
    precision_1 = dzielenieBezZera(tp, tp + fp)
    recall_1 = dzielenieBezZera(tp, tp + fn)
    f1_1 = 2 * dzielenieBezZera(precision_1 * recall_1, precision_1 + recall_1)
    support_1 = int(tp + fn)

    # dla klasy negatywnej (0)
    precision_0 = dzielenieBezZera(tn, tn + fn)
    recall_0 = dzielenieBezZera(tn, tn + fp)
    f1_0 = 2 * dzielenieBezZera(precision_0 * recall_0, precision_0 + recall_0)
    support_0 = int(tn + fp)


    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report

    report = {
        '0': {'precision': precision_0, 'recall': recall_0, 'f1-score': f1_0, 'support': support_0},
        '1': {'precision': precision_1, 'recall': recall_1, 'f1-score': f1_1, 'support': support_1},
        'accuracy': accuracy,
        'macro avg': {
            'precision': (precision_0 + precision_1) / 2,
            'recall': (recall_0 + recall_1) / 2,
            'f1-score': (f1_0 + f1_1) / 2,
            'support': support_0 + support_1
        },
        'weighted avg': {
            'precision': dzielenieBezZera(precision_0 * support_0 + precision_1 * support_1, support_0 + support_1),
            'recall': dzielenieBezZera(recall_0 * support_0 + recall_1 * support_1, support_0 + support_1),
            'f1-score': dzielenieBezZera(f1_0 * support_0 + f1_1 * support_1, support_0 + support_1),
            'support': support_0 + support_1
        }
    }
    return report

print(f"\n--- WYNIKI DLA PROGU: {threshold} ---")

print("\n--- Macierz pomyłek (scikit-learn vs manualna): ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(manual_confusion_matrix(y_test, y_pred))

print("\n--- Raport klasyfikacji (scikit-learn): ---")
print(json.dumps(classification_report(y_test, y_pred, output_dict=True), indent=2))

print("\n--- Raport klasyfikacji (manualny): ---")
print(json.dumps(manual_classification_report(y_test, y_pred, output_dict=True), indent=2))

ConfusionMatrixDisplay(cm, display_labels=data.target_names).plot()
plt.title(f"Macierz Pomyłek (próg = {threshold})")
filename = f"confusion_matrix_threshold_{threshold}.png"
plt.savefig(filename)
