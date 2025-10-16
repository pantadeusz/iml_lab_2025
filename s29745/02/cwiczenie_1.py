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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.5, help="Próg decyzji")
args = parser.parse_args()
threshold = args.threshold
print(f"Użyty próg decyzji: {threshold}")

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
#y_pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= threshold).astype(int)

# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    tn, tp, fn, fp = 0, 0, 0, 0

    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            tp += 1
        elif actual == 0 and predicted == 0:
            tn += 1
        elif actual == 0 and predicted == 1:
            fp += 1
        elif actual == 1 and predicted == 0:
            fn += 1

    return np.array([[tn, fp], [fn, tp]])


def manual_classification_report(y_true, y_pred, output_dict=True):
    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report
    cm = manual_confusion_matrix(y_true, y_pred)
    tn = int(cm[0, 0])
    fp = int(cm[0, 1])
    fn = int(cm[1, 0])
    tp = int(cm[1, 1])

    # Dla przypadków pozytywnych
    precision_1 = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall_1 = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1_score_1 = (
        2.0 * (precision_1 * recall_1) / (precision_1 + recall_1)
        if (precision_1 + recall_1) > 0.0
        else 0.0
    )
    support_1 = float(tp + fn)

    # Dla przypadków negatywnych
    precision_0 = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    recall_0 = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1_score_0 = (
        2.0 * (precision_0 * recall_0) / (precision_0 + recall_0)
        if (precision_0 + recall_0) > 0.0
        else 0.0
    )
    support_0 = float(tn + fp)

    # Miary ogólne
    total = tp + tn + fp + fn
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    total_support = support_0 + support_1

    macro_precision = float((precision_0 + precision_1) / 2.0)
    macro_recall = float((recall_0 + recall_1) / 2.0)
    macro_f1 = float((f1_score_0 + f1_score_1) / 2.0)

    if total_support > 0:
        weighted_precision = float(
            (precision_0 * support_0 + precision_1 * support_1) / total_support
        )
        weighted_recall = float(
            (recall_0 * support_0 + recall_1 * support_1) / total_support
        )
        weighted_f1 = float(
            (f1_score_0 * support_0 + f1_score_1 * support_1) / total_support
        )
    else:
        weighted_precision = weighted_recall = weighted_f1 = 0.0

    report_dict = {
        "0": {
            "precision": precision_0,
            "recall": recall_0,
            "f1-score": f1_score_0,
            "support": support_0,
        },
        "1": {
            "precision": precision_1,
            "recall": recall_1,
            "f1-score": f1_score_1,
            "support": support_1,
        },
        "accuracy": accuracy,
        "macro avg": {
            "precision": macro_precision,
            "recall": macro_recall,
            "f1-score": macro_f1,
            "support": total_support,
        },
        "weighted avg": {
            "precision": weighted_precision,
            "recall": weighted_recall,
            "f1-score": weighted_f1,
            "support": total_support,
        },
    }

    return report_dict



# Użyj scikit-learn
cm = confusion_matrix(y_test, y_pred)
print("\nGotowa macierz pomyłek")
print(cm)
cm2 = manual_confusion_matrix(y_test, y_pred)
print("\nManualna macierz pomyłek")
print(cm2)
print("\nRaport gotowy")
print(classification_report(y_test, y_pred, output_dict=True))
print("\nRaport stworzony manualnie")
print(manual_classification_report(y_test, y_pred, output_dict=True))
# Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png")
plt.close()
