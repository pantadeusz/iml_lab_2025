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
from sklearn.preprocessing import StandardScaler

# Załaduj dane
data = load_breast_cancer()
X, y = data.data, data.target
# Podziel dane
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Trenuj model
model = LogisticRegression(random_state=42, max_iter=3000)
model.fit(X_train, y_train)

# Predykcje
y_prob = model.predict_proba(X_test)[:, 1]  # prawdopodobieństwo klasy 1

# Zastosowanie progu decyzyjnego
y_pred = (y_prob >= 0.5).astype(int)


# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1)) # True Positive (Chcieliśmy 1 i mamy 1)
    TN = np.sum((y_true == 0) & (y_pred == 0)) # True Negative (Chcieliśmy 0 i mamy 0)
    FP = np.sum((y_true == 0) & (y_pred == 1)) # False Positive (Chcieliśmy 1, jednak mamy 0)
    FN = np.sum((y_true == 1) & (y_pred == 0)) # False Negative (Chcieliśmy 0, jednak mamy 1)
    return np.array([[TN, FP], [FN, TP]])
    # [TN | FP]
    # [FN | TP]


def manual_classification_report(y_true, y_pred, output_dict=True):
    cm = manual_confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    prec_0 = TN / (TN + FN) if (TN + FN) > 0 else 0
    rec_0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_0 = 2 * prec_0 * rec_0 / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0

    prec_1 = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec_1 = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_1 = 2 * prec_1 * rec_1 / (prec_1 + rec_1) if (prec_1 + rec_1) > 0 else 0

    accuracy = (TP + TN) / len(y_true)

    if output_dict:
        return {
            "0": {"precision": float(prec_0), "recall": float(rec_0),
                  "f1-score": float(f1_0), "support": float(np.sum(y_true == 0))},
            "1": {"precision": float(prec_1), "recall": float(rec_1),
                  "f1-score": float(f1_1), "support": float(np.sum(y_true == 1))},
            "accuracy": float(accuracy)
        }
    else:
        return f"""
Klasa 0: prec={prec_0:.3f}, rec={rec_0:.3f}, f1={f1_0:.3f}, sup={int(np.sum(y_true == 0))}
Klasa 1: prec={prec_1:.3f}, rec={rec_1:.3f}, f1={f1_1:.3f}, sup={int(np.sum(y_true == 1))}
Accuracy: {accuracy:.3f}
"""

    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report


# Użyj scikit-learn
cm_skl = confusion_matrix(y_test, y_pred)
print(cm_skl)
cm_manual = manual_confusion_matrix(y_test, y_pred)
print(cm_manual)
print(classification_report(y_test, y_pred, output_dict=False))
print(classification_report(y_test, y_pred, output_dict=True))
print(manual_classification_report(y_test, y_pred, output_dict=True))
# Wizualizacja
ConfusionMatrixDisplay(cm_skl).plot()
plt.savefig("confusion_matrix.png")
plt.close()