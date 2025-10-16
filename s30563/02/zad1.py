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
threshold = 0.5
if len(sys.argv) > 1:
    threshold = float(sys.argv[1])

y_pred_proba = model.predict_proba(X_test)
y_pred = (y_pred_proba[:, 1] >= threshold).astype(int)


def calculate_f1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0


# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    # Implementacja macierzy pomyłek i miar
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    # macierz standardu sklearn
    matrix_np = np.array([[tn, fp], [fn, tp]])
    matrix_np = matrix_np.astype(int)

    return matrix_np


def calculate_sensitivity_specificity(y_true, y_pred):
    cm = manual_confusion_matrix(y_true, y_pred)
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    tn = cm[0][0]

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return float(sensitivity), float(specificity)

def manual_classification_report(y_true, y_pred, output_dict=True):
    confusion_matrix = manual_confusion_matrix(y_true, y_pred)
    labels = np.unique(y_true)

    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    """
    manual_precision = confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[0][1])
    manual_recall = confusion_matrix[1][1]/(confusion_matrix[1][1] + confusion_matrix[1][0])
    manual_f1_score = 2 * manual_precision * manual_recall / (manual_precision + manual_recall)
    manual_support = y_true.size

    return f"precision: {manual_precision}, recall: {manual_recall}, f1-score: {manual_f1_score}, support: {manual_support}"
    """

    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report

    precisions = []
    recalls = []
    f1_scores = []
    supports = []

    result = {}

    for i, label in enumerate(labels):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp > 0 else 0
        recall = tp / (tp + fn) if tp + fn > 0 else 0
        f1_score = calculate_f1_score(precision, recall)
        support = confusion_matrix[i, :].sum()

        result[str(label)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1-score": float(f1_score),
            "support": int(support)
        }

        precisions.append(float(precision))
        recalls.append(float(recall))
        f1_scores.append(float(f1_score))
        supports.append(float(support))

    accuracy = float(confusion_matrix.diagonal().sum() / confusion_matrix.sum()) if confusion_matrix.sum() > 0 else 0

    macro_avg_precision = float(np.mean(precisions))
    macro_avg_recall = float(np.mean(recalls))
    macro_avg_f1_score = calculate_f1_score(macro_avg_precision, macro_avg_recall)

    weighted_avg_precision = float(np.average(precisions, weights=supports))
    weighted_avg_recall = float(np.average(recalls, weights=supports))
    weighted_avg_f1_score = calculate_f1_score(
        weighted_avg_precision, weighted_avg_recall
    )

    sum_support = float(confusion_matrix.sum())

    result["accuracy"] = accuracy

    result["macro avg"] = {
        "precision": macro_avg_precision,
        "recall": macro_avg_recall,
        "f1-score": macro_avg_f1_score,
        "support": sum_support,
    }
    result["weighted avg"] = {
        "precision": weighted_avg_precision,
        "recall": weighted_avg_recall,
        "f1-score": weighted_avg_f1_score,
        "support": sum_support,
    }

    if output_dict:
        return result

    field_width = 12
    headers = ["", "precision", "recall", "f1-score", "support"]

    report = "".join(f"{h:>{field_width}}" for h in headers) + "\n"

    for key, metrics in result.items():
        if key in ["accuracy"]:
            line = f"{key:>{field_width}}{"":>{field_width}}{"":>{field_width}}{metrics:>{field_width}.2f}{sum_support:>{field_width}}"
            report += "\n" + line + "\n"
        else:
            line = f"{key:>{field_width}}{metrics['precision']:>{field_width}.2f}{metrics['recall']:>{field_width}.2f}{metrics['f1-score']:>{field_width}.2f}{metrics['support']:>{field_width}}"
            report += line + "\n"

    return report


# Użyj scikit-learn
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm2 = manual_confusion_matrix(y_test, y_pred)
print(cm2)
print(classification_report(y_test, y_pred, output_dict=True))
print(manual_classification_report(y_test, y_pred, output_dict=True))
# print(classification_report(y_test, y_pred, output_dict=False))
# print(manual_classification_report(y_test, y_pred, output_dict=False))

# czułość i swoistość
sensitivity, specificity = calculate_sensitivity_specificity(y_test, y_pred)
print(f"Czułość: {sensitivity}")
print(f"Swoistość: {specificity}")


# Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png")
plt.close()

ConfusionMatrixDisplay(cm2).plot()
plt.savefig("manual_confusion_matrix.png")
plt.close()