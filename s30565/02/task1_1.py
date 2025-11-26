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

# parser = argparse.ArgumentParser()
# parser.add_argument("--threshold", "-t", type=float, default=0.5, help="Próg decyzyjny dla klasy 1")
# args = parser.parse_args()

# Trenuj model
model = LogisticRegression(random_state=42, max_iter=3000)
model.fit(X_train, y_train)

threshold = 0.71

proba = model.predict_proba(X_test)[:, 1]
y_pred_thr = (proba >= threshold).astype(int)

# Predykcje
y_pred = model.predict(X_test)

#sensitivity=0.986, specificity=0.907 dla threshold=0.20
# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return np.array([[tn, fp], [fn, tp]], dtype=int)

def safe_div(n,d):
    return n / d if d != 0 else 0.0

def manual_classification_report(y_true, y_pred, output_dict=True):
    confusion_matrix = manual_confusion_matrix(y_true, y_pred)
    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    precision_0 = safe_div(tn, tn + fn)
    recall_0 = safe_div(tn, tn + fp)
    f1_0 = safe_div(2 * precision_0 * recall_0, precision_0 + recall_0)

    # liczba probek w klasie 0/1 w y_true
    support1 = int(np.sum(y_true == 1))
    support0 = int(np.sum(y_true == 0))

    return {
        "0": {"precision": precision_0, "recall": recall_0, "f1-score": f1_0, "support": support0},
        "1": {"precision": precision, "recall": recall, "f1-score": f1, "support": support1},
    }




# Użyj scikit-learn
cm = confusion_matrix(y_test, y_pred_thr)
print(cm)
cm2 = manual_confusion_matrix(y_test, y_pred_thr)
print(cm2)
print(classification_report(y_test, y_pred_thr, output_dict=True))
print(manual_classification_report(y_test, y_pred_thr, output_dict=True))

tn, fp, fn, tp = cm2.ravel()
sensitivity = safe_div(tp, tp + fn)
specificity = safe_div(tn, tn + fp)
print(f"czułość={sensitivity:.3f}, swoistość={specificity:.3f} dla threshold={threshold:.2f}")

# Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig(f"confusion_matrix_t{threshold:.2f}.png")
plt.close()