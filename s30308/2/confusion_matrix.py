import numpy as np
import argparse
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


# Ustaw argument threshold
parser = argparse.ArgumentParser()
parser.add_argument("--threshold", type=float, default=0.5, help="Próg decyzji dla klasy 1")
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

# Predykcja normalna
# y_pred = model.predict(X_test)

# Predykcja probalistyczna
prob = model.predict_proba(X_test)
y_pred = (prob[:, 1] >= threshold).astype(int)  # wszystko >= 0.7 -> 1, reszta 0
print(y_pred)


# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    # Implementacja macierzy pomyłek i miar
    matrix = np.array([[0, 0], [0, 0]])


    # „True” = model miał rację (zgodność z rzeczywistością)
    # „False” = model się pomylił (niezgodność z rzeczywistością)
    # „Positive” = model przewidział wystąpienie zjawiska (np. choroba, spam, wada) -> wynik: 1
    # „Negative” = model przewidział brak zjawiska (np. zdrowy, nie-spam, produkt OK) -> wynik: 0

    for i in range(len(y_pred)):
        if y_true[i] == 0 and y_pred[i] == 0:  # TP - True Negative (Prawdziwe negatywne)
            matrix[0][0] += 1
        elif y_true[i] == 1 and y_pred[i] == 0:  # FN - False Negative (Fałszywe negatywne)
            matrix[1][0] += 1
        elif y_true[i] == 0 and y_pred[i] == 1:  # FP - False Positive (Fałszywe pozytywne)
            matrix[0][1] += 1
        else:  # TP - True Positive (Prawdziwe pozytywne)
            matrix[1][1] += 1
    return matrix


def manual_classification_report(y_true, y_pred):
    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report
    # Notatki
    # Biblioteka sklearn zamienia tp i tn miejscami ponieważ 0 uznaje za najmniejszą klasą dlatego TN jest na początku macierzy

    mcm = manual_confusion_matrix(y_true, y_pred)  # mcm - manual confusion matrix
    tn, fp, fn, tp = mcm.ravel()

    # Klasa 0
    precision_0 = tn / (tn + fn)
    recall_0 = tn / (tn + fp)
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
    support_0 = np.sum(y_true == 0)

    # Klasa 1
    precision_1 = tp / (tp + fp)
    recall_1 = tp / (tp + fn)
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
    support_1 = np.sum(y_true == 1)

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    macro_avg = {
        "precision": (precision_0 + precision_1) / 2,
        "recall": (recall_0 + recall_1) / 2,
        "f1-score": (f1_0 + f1_1) / 2,
        "support": support_0 + support_1,
    }

    weighted_avg = {
        "precision": (precision_0 * support_0 + precision_1 * support_1) / (support_0 + support_1),
        "recall": (recall_0 * support_0 + recall_1 * support_1) / (support_0 + support_1),
        "f1-score": (f1_0 * support_0 + f1_1 * support_1) / (support_0 + support_1),
        "support": support_0 + support_1,
    }

    print("Sensitivity", round(tp / (tp + fn), 2))
    print("Specificity", round(tn / (tn + fp), 2))

    return {
        "0": {"precision": precision_0, "recall": recall_0, "f1-score": f1_0, "support": support_0},
        "1": {"precision": precision_1, "recall": recall_1, "f1-score": f1_1, "support": support_1},
        "accuracy": accuracy,
        "macro avg": macro_avg,
        "weighted avg": weighted_avg
    }


# Użyj scikit-learn
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm2 = manual_confusion_matrix(y_test, y_pred)
print(cm2)

classification_report_manual = pd.DataFrame(manual_classification_report(y_test, y_pred))
classification_report_lib = pd.DataFrame(classification_report(y_test, y_pred, output_dict=True))

print(classification_report_lib)
print("\n")
print(classification_report_manual)

# # Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png")
plt.close()