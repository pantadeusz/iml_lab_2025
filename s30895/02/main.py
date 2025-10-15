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
print(type(data))
# Podziel dane
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# Trenuj model
model = LogisticRegression(random_state=42, max_iter=3000)
model.fit(X_train, y_train)
# Predykcje
y_pred = model.predict(X_test)

def manual_confusion_matrix(y_true, y_pred):
    TP, TN, FN, FP = 0,0,0,0
    for i in range(len(y_true)):
        if y_pred[i] == 0:
            if y_pred[i] == y_true[i]:
                TN += 1
            else:
                FP += 1
        else:
            if y_pred[i] == y_test[i]:
                TP += 1
            else:
                FN += 1
    return np.array([[TN,FN],[FP,TP]])
def manual_classification_report(y_true, y_pred, output_dict=True):
    confusion_matrix = manual_confusion_matrix(y_true, y_pred)
    # print(confusion_matrix[0])
    TN_1 = confusion_matrix[0][0]
    FP_1 = confusion_matrix[0][1]
    FN_1 = confusion_matrix[1][0]
    TP_1 = confusion_matrix[1][1]

    TP_0 = confusion_matrix[0][0]
    FN_0 = confusion_matrix[0][1]
    FP_0 = confusion_matrix[1][0]
    TN_0 = confusion_matrix[1][1]

    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report
    return {f"'0': precision: {calculate_precision(TP_0,FP_0)}, recall: {calculate_recall(TP_0,FN_0)}, f1_score: {calculate_f1_score(calculate_precision(TP_0,FP_0), calculate_recall(TP_0,FN_0))}, support: {TP_0 + FN_0} "
            f"'1': precision: {calculate_precision(TP_1,FP_1)}, recall: {calculate_recall(TP_1,FN_1)}, f1_score: {calculate_f1_score(calculate_precision(TP_1,FP_1), calculate_recall(TP_1,FN_1))}, support: {TP_1 + FN_1}"
            f"'macro avg': " 
            f"'weighted avg': "}

def calculate_precision(TP, FP):
    return TP/(TP+FP)

def calculate_recall(TP, FN):
    return TP/(TP+FN)

def calculate_f1_score(precision, recall):
    return 2*((precision*recall)/(precision+recall))

cm = confusion_matrix(y_test, y_pred)
print(cm)
cm2 = manual_confusion_matrix(y_test, y_pred)
print(cm2)
print(classification_report(y_test, y_pred, output_dict=True))
print(manual_classification_report(y_test, y_pred, output_dict=True))

# Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix_example.png")
plt.close()
