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
    matrix = np.zeros((2, 2), dtype=int)
    for i in range(len(y_true)):
        if y_true[i] == 0 and y_pred[i] == 0:
            matrix[0, 0] += 1  # TN
        elif y_true[i] == 0 and y_pred[i] == 1:
            matrix[0, 1] += 1  # FP
        elif y_true[i] == 1 and y_pred[i] == 0:
            matrix[1, 0] += 1  # FN
        elif y_true[i] == 1 and y_pred[i] == 1:
            matrix[1, 1] += 1  # TP
    return matrix


def manual_classification_report(y_true, y_pred, output_dict=True):
    matrix = manual_confusion_matrix(y_true, y_pred)
    # na podstawowy punkt wystarczy: precision, recall, f1-score, support
    # na dodatkowy punkt proszę zaimpementować aby wynik był taki sam jak z classification_report


    matrix = manual_confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = matrix.ravel()

    precision_0 = TN / (TN + FN) if (TN + FN) > 0 else 0.0

    recall_0 = TN / (TN + FP) if (TN + FP) > 0 else 0.0

    f1_0 = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0

    support_0 = TN + FP



    precision_1 = TP / (TP + FP) if (TP + FP) > 0 else 0.0

    recall_1 = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    f1_1 = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0

    support_1 = FN + TP

    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TN + FP + FN) > 0 else 0.0

    macro_precision = (precision_0 + precision_1) / 2
    macro_recall = (recall_0 + recall_1) / 2
    macro_f1 = (f1_0 + f1_1) / 2
    macro_support = FN + TP + TN + FP

    support_weight = support_0 + support_1
    micro_precision = (precision_0 * support_0 + precision_1 * support_1) / support_weight
    micro_recall = (recall_0 * support_0 + recall_1 * support_1) / support_weight
    micro_f1 = (f1_0 * support_0 + f1_1 * support_1) / support_weight





    return {
        '0':{
            'presision': float(precision_0),
            'recall': float(recall_0),
            'f1-score': float(f1_0),
            'support': int(support_0),
        },
        '1':{
            'precision': float(precision_1),
            'recall': float(recall_1),
            'f1-score': float(f1_1),
            'support': int(support_1),
        },
        'accuracy': float(accuracy),
        'macro-avg':{
            'precision': float(macro_precision),
            'recall': float(macro_recall),
            'f1-score': float(macro_f1),
            'support': int(macro_support),
        },
        'micro-avg':{
            'precision': float(micro_precision),
            'recall': float(micro_recall),
            'f1-score': float(micro_f1),
            'support': int(support_weight),
        }
    }


parser = argparse.ArgumentParser()
parser.add_argument("-prog", type=float, default=0.5)
args = parser.parse_args()

prog = args.prog

y_prob = model.predict_proba(X_test)[:,1]
y_pred_prob = np.where(y_prob >= prog, 1, 0)

print(f"Próg: {prog}")
print(manual_classification_report(y_test, y_pred_prob))






# Użyj scikit-learn
cm = confusion_matrix(y_test, y_pred)
print(cm)
cm2 = manual_confusion_matrix(y_test, y_pred)
print(cm2)
print(classification_report(y_test, y_pred, output_dict=True))
print(manual_classification_report(y_test, y_pred, output_dict=True))



# Wizualizacja
ConfusionMatrixDisplay(cm).plot()
plt.savefig("confusion_matrix.png")
plt.close()
