from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.metrics import classification_report
import numpy as np


def prepare_data():
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Oversampling
    X_smote, y_smote = SMOTE(random_state=42).fit_resample(X_train, y_train)

    # Undersampling
    X_under, y_under = RandomUnderSampler(random_state=42).fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test, X_smote, y_smote, X_under, y_under


def train_models(X_train, y_train, X_smote, y_smote, X_under, y_under):
    models = {
        'base': LogisticRegression(random_state=42).fit(X_train, y_train),
        'under': LogisticRegression(random_state=42).fit(X_under, y_under),
        'smote': LogisticRegression(random_state=42).fit(X_smote, y_smote),
        'balanced': LogisticRegression(random_state=42, class_weight='balanced').fit(X_train, y_train)
    }
    return models


def evaluate_models(models, X_test, y_test, output_frame='compare_models.png'):
    plt.figure(figsize=(12, 5))

    for name, model in models.items():
        # ROC
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:0.2f})')

        # Precision/Recall
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_auc = auc(recall, precision)

        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'{name} (AUC={pr_auc:0.2f})')

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
    plt.savefig(output_frame)


def treshhold_check(threshold, models, X_test, y_test):
    print(f"\nThreshold = {threshold}")
    for name, model in models.items():

        y_score = model.predict_proba(X_test)[:, 1]
        y_pred = np.where(y_score >= threshold, 1, 0)

        print(f"\nModel: {name}")
        print(classification_report(y_test, y_pred, digits=3))

def main():
    X_train, X_test, y_train, y_test, X_smote, y_smote, X_under, y_under = prepare_data()
    models = train_models(X_train, y_train, X_smote, y_smote, X_under, y_under)
    evaluate_models(models, X_test, y_test)

    treshhold_check(0.5, models, X_test, y_test)
    treshhold_check(0.2, models, X_test, y_test)
    treshhold_check(0.8, models, X_test, y_test)


if __name__ == '__main__':
    main()
