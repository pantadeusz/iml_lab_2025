import numpy as np
import argparse

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
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(y_true.size):
        if y_true[i] == 0:
            if y_pred[i] == 0:
                tn += 1
            else:
                fp += 1
        else:
            if y_pred[i] == 0:
                fn += 1
            else:
                tp += 1

    return np.matrix([[tn, fp], [fn, tp]])


def manual_classification_report(y_true, y_pred, output_dict=True):
    confusion_matrix = manual_confusion_matrix(y_true, y_pred)
    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report
    precision_1 = confusion_matrix[1, 1] / (
        confusion_matrix[1, 1] + confusion_matrix[0, 1]
    )
    precision_0 = confusion_matrix[0, 0] / (
        confusion_matrix[0, 0] + confusion_matrix[1, 0]
    )
    f1_1 = (2 * confusion_matrix[1, 1]) / (
        (2 * confusion_matrix[1, 1]) + confusion_matrix[1, 0] + confusion_matrix[0, 1]
    )
    f1_0 = (2 * confusion_matrix[0, 0]) / (
        (2 * confusion_matrix[0, 0]) + confusion_matrix[0, 1] + confusion_matrix[1, 0]
    )
    recall_1 = confusion_matrix[1, 1] / (
        confusion_matrix[1, 1] + confusion_matrix[1, 0]
    )
    recall_0 = confusion_matrix[0, 0] / (
        confusion_matrix[0, 0] + confusion_matrix[0, 1]
    )
    specificity_0 = confusion_matrix[1, 1] / (
        confusion_matrix[1, 1] + confusion_matrix[1, 0]
    )
    specificity_1 = confusion_matrix[0, 0] / (
        confusion_matrix[0, 0] + confusion_matrix[0, 1]
    )
    support_0 = 0
    support_1 = 0
    accuracy = (confusion_matrix[0, 0] + confusion_matrix[1, 1]) / (
        confusion_matrix[0, 0]
        + confusion_matrix[1, 1]
        + confusion_matrix[0, 1]
        + confusion_matrix[1, 0]
    )

    for i in y_true:
        if i == 0:
            support_0 += 1
        else:
            support_1 += 1

    w_0 = support_0 / y_true.size
    w_1 = 1 - w_0

    precision_macro = 0.5 * precision_0 + 0.5 * precision_1
    f1_macro = 0.5 * f1_0 + 0.5 * f1_1
    recall_macro = 0.5 * recall_0 + 0.5 * recall_1

    precision_weighted = w_0 * precision_0 + w_1 * precision_1
    f1_weighted = w_0 * f1_0 + w_1 * f1_1
    recall_weighted = w_0 * recall_0 + w_1 * recall_1

    report = {
        "0": {
            "precision": float(precision_0),
            "recall": float(recall_0),
            "f1-score": float(f1_0),
            "support": float(support_0),
            "specificity": float(specificity_0),
        },
        "1": {
            "precision": float(precision_1),
            "recall": float(recall_1),
            "f1-score": float(f1_1),
            "support": float(support_1),
            "specificity": float(specificity_1),
        },
        "accuracy": float(accuracy),
        "macro avg": {
            "precision": float(precision_macro),
            "recall": float(recall_macro),
            "f1-score": float(f1_macro),
            "support": float(y_true.size),
        },
        "weighted avg": {
            "precision": float(precision_weighted),
            "recall": float(recall_weighted),
            "f1-score": float(f1_weighted),
            "support": float(y_true.size),
        },
    }
    return report


def decision_function(X_test, threshold):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    return y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    y_proba = model.predict_proba(X_test)
    y_pred_with_custom_threshold = decision_function(X_test, args.threshold)

    # Użyj scikit-learn
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    cm2 = manual_confusion_matrix(y_test, y_pred)
    print(cm2)
    print(classification_report(y_test, y_pred, output_dict=True))
    print(manual_classification_report(y_test, y_pred, output_dict=True))
    print(
        manual_classification_report(
            y_test, y_pred_with_custom_threshold, output_dict=True
        )
    )
    # Wizualizacja
    ConfusionMatrixDisplay(cm2).plot()
    plt.savefig("confusion_matrix_manual.png")
    plt.close()
