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
# Predykcje
y_pred = model.predict(X_test)


# Ręczne obliczenie miar
def manual_confusion_matrix(y_true, y_pred):
    # Implementacja macierzy pomyłek i miar
    matrix = [[0, 0], [0, 0]]

    for i in range(len(y_pred)):
        if int(y_true[i]) == 1 and int(y_pred[i]) == 1: # TP - True Positive
            matrix[0, 0] += 1
        if int(y_true[i]) == 0 and int(y_pred[i]) == 1: # FN - False Negative
            matrix[1, 0] += 1
        if int(y_true[i]) == 1 and int(y_pred[i]) == 0: # FP - False Positive
            matrix[0, 1] += 1
        if int(y_true[1]) == 1 and int(y_pred[i]) == 0: # TP - True Positive
            matrix[1, 1] += 1
    print(matrix)
    return matrix

def manual_classification_report(y_true, y_pred, output_dict=True):
    confusion_matrix = manual_confusion_matrix(y_true, y_pred)
    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report
    pass



# Użyj scikit-learn
print(manual_classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
print(cm)
# cm2 = manual_confusion_matrix(y_test, y_pred)
# print(cm2)
# print(classification_report(y_test, y_pred, output_dict=True))
# print(manual_classification_report(y_test, y_pred, output_dict=True))
# # Wizualizacja
# ConfusionMatrixDisplay(cm).plot()
# plt.savefig("confusion_matrix.png")
# plt.close()