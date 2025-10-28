import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay

# parser
p = argparse.ArgumentParser()
p.add_argument("--threshold", type=float, default=0.5)
threshold = p.parse_args().threshold

# dane i podzial
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# skalowanie i regresja logistyczna
model = make_pipeline(StandardScaler(), LogisticRegression(random_state=42, max_iter=3000))
model.fit(X_train, y_train)

# predykcje
proba = model.predict_proba(X_test)[:, 1]
y_pred = (proba >= threshold).astype(int)

# macierz pomylek
def cm_manual(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    return np.array([[tn, fp], [fn, tp]])

# raport
def report_manual(y_true, y_pred):
    tn, fp, fn, tp = cm_manual(y_true, y_pred).ravel()
    sd = lambda a, b: a / b if b else 0.0
    prec1 = sd(tp, tp + fp); rec1 = sd(tp, tp + fn); f1_1 = sd(2 * prec1 * rec1, prec1 + rec1); sup1 = tp + fn
    prec0 = sd(tn, tn + fn); rec0 = sd(tn, tn + fp); f1_0 = sd(2 * prec0 * rec0, prec0 + rec0); sup0 = tn + fp
    total = sup0 + sup1; acc = sd(tp + tn, total)
    return {
        "0": {"precision": prec0, "recall": rec0, "f1": f1_0, "support": int(sup0)},
        "1": {"precision": prec1, "recall": rec1, "f1": f1_1, "support": int(sup1)},
        "accuracy": acc,
        "macro_avg": {"precision": (prec0 + prec1) / 2, "recall": (rec0 + rec1) / 2, "f1": (f1_0 + f1_1) / 2},
        "weighted_avg": {
            "precision": sd(prec0 * sup0 + prec1 * sup1, total),
            "recall": sd(rec0 * sup0 + rec1 * sup1, total),
            "f1": sd(f1_0 * sup0 + f1_1 * sup1, total),
        },
    }

# print i zapis wykresu
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred, output_dict=True))
print(report_manual(y_test, y_pred))
ConfusionMatrixDisplay(cm).plot(); plt.savefig("confusion_matrix.png"); plt.close()