import argparse
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

parser = argparse.ArgumentParser()
parser.add_argument(
    "--threshold", type=float, default=0.5, help="Próg decyzyjny (default: 0.5)"
)
args = parser.parse_args()
threshold = args.threshold

# Załaduj dane
data = load_breast_cancer()
X, y = data.data, data.target
# Podziel dane
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Trenuj model
model = LogisticRegression(random_state=42, max_iter=4000)
model.fit(X_train, y_train)
# Predykcje
y_pred = model.predict(X_test)


# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    # Implementacja macierzy pomyłek i miar
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_true[i] == 1 and y_pred[i] == 1:
            TP += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        else:
            TN += 1
    return np.array([[TN, FP], [FN, TP]])


def manual_classification_report(y_true, y_pred, output_dict=True):
    mcm = manual_confusion_matrix(y_true, y_pred)
    TN, FP = mcm[0]
    FN, TP = mcm[1]

    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    precision_0 = TN / (TN + FN)
    recall_0 = TN / (TN + FP)
    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
    support_0 = TN + FP

    precision_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
    support_1 = TP + FN

    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report
    accuracy = (TP + TN) / (support_0 + support_1)

    total_support = support_0 + support_1

    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2

    weighted_precision = (
        precision_0 * support_0 + precision_1 * support_1
    ) / total_support
    weighted_recall = (recall_0 * support_0 + recall_1 * support_1) / total_support
    weighted_f1 = (f1_0 * support_0 + f1_1 * support_1) / total_support

    if output_dict:
        return {
            "0": {
                "precision": float(precision_0),
                "recall": float(recall_0),
                "f1-score": float(f1_0),
                "support": float(support_0),
            },
            "1": {
                "precision": float(precision_1),
                "recall": float(recall_1),
                "f1-score": float(f1_1),
                "support": float(support_1),
            },
            "accuracy": float(accuracy),
            "macro avg": {
                "precision": float(macro_precision),
                "recall": float(macro_recall),
                "f1-score": float(macro_f1),
                "support": float(total_support),
            },
            "weighted avg": {
                "precision": float(weighted_precision),
                "recall": float(weighted_recall),
                "f1-score": float(weighted_f1),
                "support": float(total_support),
            },
        }


y_proba = model.predict_proba(X_test)[:, 1]
y_proba_pred = (y_proba >= threshold).astype(int)

print(
    f"Próg decyzyjny: {threshold}\n"
    f"{manual_classification_report(y_test, y_proba_pred)}"
)

# Użyj scikit-learn
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm2 = manual_confusion_matrix(y_test, y_pred)
print(cm2)
print(classification_report(y_test, y_pred, output_dict=True))
print("\n")
print(manual_classification_report(y_test, y_pred, output_dict=True))
# Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png")
ConfusionMatrixDisplay(cm2).plot()
plt.savefig("confusion_matrix2.png")
plt.close()
