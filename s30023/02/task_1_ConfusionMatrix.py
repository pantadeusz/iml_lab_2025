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
    X, y, test_size=0.2, random_state=42)
# Trenuj model
model = LogisticRegression(random_state=42, max_iter=5000)
model.fit(X_train, y_train)
# Predykcje
y_pred = model.predict(X_test)


# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    TP, FP, TN, FN = 0, 0, 0, 0
    for i in range(len(y_pred)):
        if y_true[i] == y_pred[i] == 1:
            TP += 1
        if y_pred[i] == 1 and y_true[i] != y_pred[i]:
            FP += 1
        if y_true[i] == y_pred[i] == 0:
            TN += 1
        if y_pred[i] == 0 and y_true[i] != y_pred[i]:
            FN += 1
    return np.array([[TN, FP],[FN, TP]])



def manual_classification_report(y_true, y_pred, output_dict=True):
    confusion_matrix = manual_confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = confusion_matrix[0, 0], confusion_matrix[0, 1], confusion_matrix[1, 0], confusion_matrix[1, 1]

    precision_0 = TN / (TN + FN)
    recall_0 = TN / (TN + FP)
    f1_score_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0)
    support_0 = TN + FP

    precision_1 = TP / (TP + FP)
    recall_1 = TP / (TP + FN)
    f1_score_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1)
    support_1 = FN + TP

    accuracy = (TP + TN) / (TP + TN + FP + FN)

    precision_macro = (precision_0 + precision_1) / 2
    recall_macro = (recall_0 + recall_1) / 2
    f1_macro = (f1_score_0 + f1_score_1) / 2
    support_macro = support_0 + support_1

    support_weighted = support_0 + support_1
    precision_weighted = (precision_0 * support_0 + precision_1 * support_1) / support_weighted
    recall_weighted = (recall_0 * support_0 + recall_1 * support_1) / support_weighted
    f1_weighted = (f1_score_0 * support_0 + f1_score_1 * support_1) / support_weighted

    if output_dict:
        return {
            "0": {
                "precision": float(precision_0),
                "recall": float(recall_0),
                "f1-score": float(f1_score_0),
                "support": float(support_0),
            },
            "1": {
                "precision": float(precision_1),
                "recall": float(recall_1),
                "f1-score": float(f1_score_1),
                "support": float(support_1),
            },
            "accuracy": float(accuracy),
            "macro avg":{
                "precision": float(precision_macro),
                "recall": float(recall_macro),
                "f1-score": float(f1_macro),
                "support": float(support_macro),
            },
            "weighted avg":{
                "precision": float(precision_weighted),
                "recall": float(recall_weighted),
                "f1-score": float(f1_weighted),
                "support": float(support_weighted),
            }
        }


# Użyj scikit-learn
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm2 = manual_confusion_matrix(y_test, y_pred)
print(cm2)
print(classification_report(y_test, y_pred, output_dict=True))
print(manual_classification_report(y_test, y_pred, output_dict=True))
# Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix_cm.png")
ConfusionMatrixDisplay(cm2).plot()
plt.savefig("confusion_matrix_cm2.png")
plt.close()