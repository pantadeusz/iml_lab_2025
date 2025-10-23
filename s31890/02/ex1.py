import numpy as np
import sys

from sklearn.datasets import load_breast_cancer  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import (
    train_test_split,
)  # pyright: ignore[reportMissingImports]
from sklearn.linear_model import (
    LogisticRegression,
)  # pyright: ignore[reportMissingImports]
from sklearn.metrics import (  # pyright: ignore[reportMissingImports]
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]

# Załaduj dane
data = load_breast_cancer()
X, y = data.data, data.target  # Podziel dane
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Trenuj model
model = LogisticRegression(random_state=42, max_iter=3000)
model.fit(X_train, y_train)


# Prawdopodobności
proba = model.predict_proba(X_test)

# Treshold from cmd
if not len(sys.argv) == 2:
    raise Exception("Wrong number of arguments!")
threshold = float(sys.argv[1])
print(f"Threshold set as: {threshold}")

# Predykcje
# y_pred = model.predict(X_test, threshold=threshold)
y_pred = (proba[:, 1] >= threshold).astype(int)

# Drukuj porównanie
for i, (pred, prob) in enumerate(zip(y_pred, proba)):
    print(
        f"Sample {i + 1}: Predicted Class = {pred}, Class 0 probability = {prob[0]:.4f}, Class 1 probability = {prob[1]:.4f}"
    )


# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    matrix = np.zeros((2, 2))
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i] and int(y_true[i]) == 1:
            matrix[0, 0] += 1
        elif y_true[i] != y_pred[i] and int(y_true[i]) == 1:
            matrix[0, 1] += 1
        elif y_true[i] == y_pred[i] and int(y_true[i]) == 0:
            matrix[1, 1] += 1
        elif y_true[i] != y_pred[i] and int(y_true[i]) == 0:
            matrix[1, 0] += 1
        else:
            raise ValueError("Invalid input")

    return matrix


def manual_classification_report(y_true, y_pred, output_dict=True):
    confusion_matrix = manual_confusion_matrix(y_true, y_pred)
    precision = confusion_matrix[0, 0] / (
        confusion_matrix[0, 0] + confusion_matrix[0, 1]
    )
    recall = confusion_matrix[0, 0] / (confusion_matrix[0, 0] + confusion_matrix[1, 0])
    f1_score = 2 * precision * recall / (precision + recall)
    support = confusion_matrix[0, 0] + confusion_matrix[0, 1]

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1-score": float(f1_score),
        "support": float(support),
    }
    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report


# Użyj scikit-learn
print("Scikit-learn:\n")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("Mine:\n")
cm2 = manual_confusion_matrix(y_test, y_pred)
print(cm2)
print("Scikit-learn:\n")
print(classification_report(y_test, y_pred, output_dict=True))
print("Mine:\n")
print(manual_classification_report(y_test, y_pred, output_dict=True))
# Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png")
plt.close()

ConfusionMatrixDisplay(cm2).plot()
plt.savefig("confusion_matrix_manual.png")
plt.close()
