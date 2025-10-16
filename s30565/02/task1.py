import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt

data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LogisticRegression(random_state=42, max_iter=3000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def manual_classification_report(y_true, y_pred, output_dict=True):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    report = {}
    classes = np.unique(y_true)

    for cls in classes:
        TP = np.sum((y_true == cls) & (y_pred == cls))
        FP = np.sum((y_true != cls) & (y_pred == cls))
        FN = np.sum((y_true == cls) & (y_pred != cls))
        support = np.sum(y_true == cls)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        report[cls] = {
            "precision": precision,
            "recall": recall,
            "f1-score": f1,
            "support": support
        }

    precisions = [report[cls]["precision"] for cls in classes]
    recalls = [report[cls]["recall"] for cls in classes]
    f1s = [report[cls]["f1-score"] for cls in classes]
    supports = [report[cls]["support"] for cls in classes]
    total_support = np.sum(supports)

    macro_avg = {
        "precision": np.mean(precisions),
        "recall": np.mean(recalls),
        "f1-score": np.mean(f1s),
        "support": total_support
    }

    weighted_avg = {
        "precision": np.average(precisions, weights=supports),
        "recall": np.average(recalls, weights=supports),
        "f1-score": np.average(f1s, weights=supports),
        "support": total_support
    }

    report["macro avg"] = macro_avg
    report["weighted avg"] = weighted_avg

    if output_dict:
        return report
    else:
        print(f"{'Class':<10}{'Precision':>10}{'Recall':>10}{'F1-Score':>10}{'Support':>10}")
        for cls in classes:
            vals = report[cls]
            print(f"{cls:<10}{vals['precision']:>10.2f}{vals['recall']:>10.2f}{vals['f1-score']:>10.2f}{vals['support']:>10}")
        print(f"{'macro avg':<10}{macro_avg['precision']:>10.2f}{macro_avg['recall']:>10.2f}{macro_avg['f1-score']:>10.2f}{macro_avg['support']:>10}")
        print(f"{'weighted avg':<10}{weighted_avg['precision']:>10.2f}{weighted_avg['recall']:>10.2f}{weighted_avg['f1-score']:>10.2f}{weighted_avg['support']:>10}")

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
