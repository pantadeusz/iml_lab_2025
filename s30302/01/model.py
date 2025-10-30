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
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--threshold', type=float, default=0.5, help='Próg decyzyjny')
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
model = LogisticRegression(random_state=42, max_iter=5000)
model.fit(X_train, y_train)
# Predykcje
y_pred = model.predict(X_test)


# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            TP += 1
        if true == 0 and pred == 0:
            TN += 1
        if pred == 1 and true == 0:
            FP += 1
        if pred == 0 and true == 1:
            FN += 1
    return np.array([[TN, FP],
                     [FN, TP]])


def manual_classification_report(y_true, y_pred, output_dict=True):
    confusion_matrix = manual_confusion_matrix(y_true, y_pred)
    precision_1 = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[0, 1])
    recall_1 = confusion_matrix[1, 1] / (confusion_matrix[1, 1] + confusion_matrix[1, 0])
    f1_1 = 2 * (precision_1 * recall_1) / (precision_1 + recall_1)
    support_1 = confusion_matrix[1, 0] + confusion_matrix[1, 1]

    precision_0 = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
    recall_0 = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[0, 1])
    f1_0 = 2 * (precision_0 * recall_0) / (precision_0 + recall_0)
    support_0 = confusion_matrix[0, 0] + confusion_matrix[0, 1]

    accuracy = ((confusion_matrix[1, 1] + confusion_matrix[0, 0]) / confusion_matrix.sum())

    raport = {
        '0': {
            'precision': precision_0,
            'recall': recall_0,
            'f1-score': f1_0,
            'support': support_0,
        },
        '1': {
            'precision': precision_1,
            'recall': recall_1,
            'f1-score': f1_1,
            'support': support_1,
        },
        'macro avg': {
            'precision': (precision_0 + precision_1) / 2,
            'recall': (recall_0 + recall_1) / 2,
            'f1-score': (f1_0 + f1_1) / 2,
            'support': support_0 + support_1
        },
        'weighted avg': {
        'precision': (precision_0 * support_0 + precision_1 * support_1) / (support_0 + support_1),
        'recall': (recall_0 * support_0 + recall_1 * support_1) / (support_0 + support_1),
        'f1-score': (f1_0 * support_0 + f1_1 * support_1) / (support_0 + support_1),
        'support': support_0 + support_1
        },
        'accuracy': accuracy
    }
    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report
    return raport

def predict_with_threshold(model, X, threshold=0.5):
    probs = model.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int)

y_pred_custom = predict_with_threshold(model, X_test, threshold)

cm_custom = manual_confusion_matrix(y_test, y_pred_custom)
TN, FP, FN, TP = cm_custom.ravel()
sensitivity = TP / (TP + FN)
specificity = TN / (TN + FP)

print(f"Threshold: {threshold:.2f}")
print(f"Sensitivity (Recall): {sensitivity:.2f}")
print(f"Specificity: {specificity:.2f}")

md_file = "threshold_summary.md"
with open(md_file, "w") as f:
    f.write(f"Eksperyment z progiem decyzyjnym = {threshold:.2f}\n")
    f.write(f"- Sensitivity (Recall): {sensitivity:.2f}\n")
    f.write(f"- Specificity: {specificity:.2f}\n")
    f.write("Wnioski\n")
    f.write("- Nizszy prog zwieksza czulosc, ale zmniejsza swoistosc.\n")
    f.write("- Wyzszy prog poprawia swoistosc, ale zmniejsza czulosc.\n")
    f.write("- Wybor progu zalezy od priorytetu zadania.\n")

# Użyj scikit-learn
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm2 = manual_confusion_matrix(y_test, y_pred)
print(cm2)
print(classification_report(y_test, y_pred, output_dict=True))
print(manual_classification_report(y_test, y_pred, output_dict=True))
# # Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png")
plt.close()