from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import numpy as np


def generate_data():
    X, y = make_classification(
        n_samples=1000, n_features=20,
        n_informative=10, n_redundant=5,
        n_classes=2, weights=[0.95, 0.05],
        random_state=42)
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def evaluate_models(X_train, X_test, y_train, y_test):
    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)
    X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

    models = {
        'Base': LogisticRegression(random_state=42).fit(X_train, y_train),
        'Weighted': LogisticRegression(class_weight='balanced', random_state=42).fit(X_train, y_train),
        'SMOTE': LogisticRegression(random_state=42).fit(X_smote, y_smote),
        'Under': LogisticRegression(random_state=42).fit(X_under, y_under)
    }

    # ROC
    plt.figure(figsize=(15, 5))
    for name, model in models.items():
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

    # PR
    for name, model in models.items():
        y_score = model.predict_proba(X_test)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall, precision)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')

    # Threshholds
    thresholds_to_check = [0.1, 0.3, 0.5, 0.7, 0.8]
    print("\nPorównanie wyników dla różnych progów (z metrykami):")
    for name, model in models.items():
        y_score = model.predict_proba(X_test)[:, 1]
        print(f"\nModel: {name}")
        for t in thresholds_to_check:
            y_pred = (y_score >= t).astype(int)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            pos_rate = np.mean(y_pred)
            print(f"\tPróg = {t:.1f} -> pozytywnych prognoz: {pos_rate * 100:.2f}%, "
                  f"Precision = {prec:.2f}, Recall = {rec:.2f}, F1 = {f1:.2f}")

    plt.subplot(1, 2, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    X_train, X_test, y_train, y_test = generate_data()
    evaluate_models(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()