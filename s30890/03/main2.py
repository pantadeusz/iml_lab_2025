import warnings
from sklearn.exceptions import UndefinedMetricWarning

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# generowanie sztucznych danych z niezbalansowanymi klasami
X, y = make_classification(
    n_samples=2000,
    n_features=20,
    n_informative=5,
    n_redundant=2,
    n_classes=2,
    weights=[0.95, 0.05],
    random_state=42,
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

print("Udział klasy mniejszości:", np.mean(y_train == 1))


# funkcja do ewaluacji
def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    auc = roc_auc_score(y_test, y_proba)

    return {
        "Model": name,
        "Accuracy": report["accuracy"],
        "Precision": report["1"]["precision"],
        "Recall": report["1"]["recall"],
        "F1": report["1"]["f1-score"],
        "ROC_AUC": auc,  # krzywa pokazujaca zaleznosc miedzy tpr a fpr dla roznych progow, lepszy do rozpoznawania klas niz accuracy
    }


# strategie radzenia sobie z niezbalansowanymi klasami

# a. bazowy model bez żadnej modyfikacji
base_model = LogisticRegression(max_iter=1000, random_state=42)

# b. model z ważeniem klas
weighted_model = LogisticRegression(
    max_iter=1000, class_weight="balanced", random_state=42
)

# c. oversampling za pomocą SMOTE, duplikowanie lub generowanie syntetycznych próbek klasy mniejszościowej
smote = SMOTE(random_state=42)
X_train_over, y_train_over = smote.fit_resample(X_train, y_train)

smote_model = LogisticRegression(max_iter=1000, random_state=42)

# d. undersampling za pomocą RandomUnderSampler, zmnijeszenie próbek z klasy większościowej
under = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = under.fit_resample(X_train, y_train)

under_model = LogisticRegression(max_iter=1000, random_state=42)

# ewaluacja wszystkich modeli
results = []
results.append(evaluate_model("Bazowy", base_model, X_train, y_train, X_test, y_test))
results.append(
    evaluate_model("Ważenie klas", weighted_model, X_train, y_train, X_test, y_test)
)
results.append(
    evaluate_model("SMOTE", smote_model, X_train_over, y_train_over, X_test, y_test)
)
results.append(
    evaluate_model(
        "Undersampling", under_model, X_train_under, y_train_under, X_test, y_test
    )
)

df = pd.DataFrame(results)
print("\nWyniki")
print(df)

# wizualizacja wyników
df.set_index("Model")[["Precision", "Recall", "F1", "ROC_AUC"]].plot(
    kind="bar", figsize=(10, 6)
)
plt.title("Porównanie technik dla niezbalansowanych klas")
plt.ylabel("Wartość metryki")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.show()

# analiza progu decyzyjnego
thresholds = np.linspace(0.1, 0.9, 9)
precisions, recalls = [], []

model = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]

for t in thresholds:
    y_pred_t = (y_proba > t).astype(int)
    report = classification_report(y_test, y_pred_t, output_dict=True)
    precisions.append(report["1"]["precision"])
    recalls.append(report["1"]["recall"])

plt.figure(figsize=(8, 5))
plt.plot(thresholds, precisions, marker="o", label="Precision")
plt.plot(thresholds, recalls, marker="o", label="Recall")
plt.xlabel("Próg decyzyjny")
plt.ylabel("Wartość metryki")
plt.title("Wpływ progu na Precision i Recall")
plt.legend()
plt.grid(True)
plt.savefig("threshold_analysis.png")
plt.close()

# index metryk do wykresu
metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]

# wykres porownania metryk
df_plot = df.set_index("Model")[metrics_to_plot]
df_plot.plot(kind="bar", figsize=(10, 6))

plt.title("Porównanie metryk dla różnych technik balansowania klas")
plt.ylabel("Wartość metryki")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.legend(title="Metryka")
plt.savefig("balanced_class_metrics.png")
plt.close()
