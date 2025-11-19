from sklearn.ensemble import RandomForestClassifier  # pyright: ignore[reportMissingImports]
from sklearn.svm import SVC  # pyright: ignore[reportMissingImports]
from sklearn.metrics import roc_curve, auc, precision_recall_curve  # pyright: ignore[reportMissingImports]
from sklearn.linear_model import LogisticRegression  # pyright: ignore[reportMissingImports]
import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
from sklearn.datasets import load_breast_cancer  # pyright: ignore[reportMissingImports]
from sklearn.model_selection import train_test_split  # pyright: ignore[reportMissingImports]

models = [
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("Random Forest", RandomForestClassifier(random_state=42)),
    ("SVM", SVC(probability=True, random_state=42)),
]

# Załaduj dane
data = load_breast_cancer()
X, y = data.data, data.target
# Podziel dane
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

plt.figure(figsize=(12, 5))

for name, model in models:
    model.fit(X_train, y_train)
    y_proba = model.predict_proba(X_test)[:, 1]

    # ROC
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.2f})")

plt.subplot(1, 2, 1)
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.subplot(1, 2, 2)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()

plt.savefig("precision_recall_curve.png")
plt.close()
