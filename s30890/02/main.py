import warnings
from sklearn.exceptions import ConvergenceWarning
# ignorowanie ostzezen o zbieznosci dla czytelnosci wynikow
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import argparse
# parsing argumentow z linii komend - pozwala ustawic wlasny prog decyzyjny
parser = argparse.ArgumentParser(description="binary classification threshold experiment")
parser.add_argument("--threshold", type=float, default=0.5,
                    help="decision threshold for logistic regression default = 0.5")
args = parser.parse_args()
threshold = args.threshold


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
import pandas as pd

# wczytanie gotowego zbioru danych
data = load_breast_cancer()
X, y = data.data, data.target

# podzial na zbior treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# tworzenie modelu regresji logistycznej
model = LogisticRegression(random_state=42, max_iter=3000)

# trenowanie modelu
model.fit(X_train, y_train)

# predykcje prawdopodobienstwa klasy pozytywnej
y_proba = model.predict_proba(X_test)[:, 1]

# zastosowanie progu decyzyjnego z linii komend
y_pred = (y_proba >= threshold).astype(int)


# reczne obliczanie macierzy pomyłek
def manual_confusion_matrix(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))
    return np.array([[TN, FP], [FN, TP]])


# recznie obliczenie podstawowych miar klasyfikacji binarnej
def manual_classification_report(y_true, y_pred, output_dict=True):
    cm = manual_confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()

    # obliczanie metryk
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0  # czulosc
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0 # swoistosc
    npv = TN / (TN + FN) if (TN + FN) > 0 else 0
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )

    if output_dict:
        return {
            "confusion_matrix": cm.tolist(),
            "precision": precision,
            "recall (sensitivity)": recall,
            "specificity": specificity,
            "NPV": npv,
            "accuracy": accuracy,
            "f1_score": f1,
        }
    else:
        # wersja wypisujaca miary w konsoli, nieuzywane raczej do szybkiego podgladu, testowania
        print("Macierz pomyłek: \n")
        print(cm)
        print(f"Precision: {precision:.3f} \n")
        print(f"Recall (Sensitivity): {recall:.3f} \n")
        print(f"Specificity: {specificity:.3f} \n")
        print(f"NPV: {npv:.3f} \n")
        print(f"Accuracy: {accuracy:.3f} \n")
        print(f"F1-score: {f1:.3f} \n")

# testowanie progow decyzyjnych od 0.1 do 0.9
thresholds = np.linspace(0.1, 0.9, 9)
results = []

for t in thresholds:
    # klasyfikacja przy danym progu
    y_pred_t = (y_proba >= t).astype(int)
    cm = manual_confusion_matrix(y_test, y_pred_t)
    TN, FP, FN, TP = cm.ravel()
    
    # obliczanie czulosci, swoistosci i dokladnosci
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    results.append({
        "threshold": t,
        "recall (sensitivity)": recall,
        "specificity": specificity,
        "accuracy": accuracy
    })

df_results = pd.DataFrame(results).round(3)

# wyniki / porownanie recznego i scikit-learn / wyniki dla zadanego progu
print("\n confusion matrix (scikit-learn)")
cm = confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

print("\n confusion matrix (scikit-learn)")
cm2 = manual_confusion_matrix(y_test, y_pred)
print(pd.DataFrame(cm2, index=["Actual 0", "Actual 1"], columns=["Pred 0", "Pred 1"]))

print("\n classification report (scikit-learn) ")
report_sklearn = classification_report(y_test, y_pred, output_dict=True)
print(pd.DataFrame(report_sklearn).T.round(3))

print("\n classification report (manual)")
report_manual = manual_classification_report(y_test, y_pred, output_dict=True)
report_manual_df = (
    pd.DataFrame(report_manual.items(), columns=["Metric","Value"])
    .set_index("Metric")
    .round(3)
)
print(report_manual_df)

print("\n wyniki eksperymentu wplywu progu decyzyjnego:\n")
print(df_results.to_string(index=False))


# wykres macierzy pomylek
disp = ConfusionMatrixDisplay(cm, display_labels=data.target_names)
disp.plot(cmap="Purples")
plt.title("confusion matrix (Logistic Regression)")
plt.savefig("confusion_matrix.png")
plt.close()

# wykres wplywu progu na czulosc i swoistosc
plt.plot(df_results["threshold"], df_results["recall (sensitivity)"], label="czulosc (Recall)", color='purple', linestyle='-')
plt.plot(df_results["threshold"], df_results["specificity"], label="swoistosc (Specificity)", color='pink', linestyle='-')
plt.style.use("ggplot")
plt.xlabel("prog decyzyjny")
plt.ylabel("wartosc metryki")
plt.title("wplyw progu na czulosc i swoistosc")
plt.legend()
plt.grid(True, which='both', linestyle="--", linewidth=0.5)
plt.savefig("threshold_experiment.png", dpi=150)
plt.close()
