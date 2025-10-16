from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def get_X_y():
    data = load_diabetes()
    X, y = data.data, data.target

    y_prog_of_illness = (y > np.median(y)).astype(int)

    return X, y_prog_of_illness

def MCAR_data(X):
    rng = np.random.RandomState(42)
    missing_mask = rng.rand(*X.shape) < 0.1  # 10% braków
    X_missing = X.copy()
    X_missing[missing_mask] = np.nan

    return X_missing

def imputers_analyze_model(imputers, X, y):
    plt.figure(figsize=(15, 5))
    for name, imputer in imputers.items():
        X_imputed = imputer.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y,
                                                            test_size=0.2, random_state=42)

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_score = model.predict(X_test)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        # PR
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        pr_auc = auc(recall, precision)
        plt.subplot(1, 2, 2)
        plt.plot(recall, precision, label=f'{name} (AUC = {pr_auc:.2f})')

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

    plt.show()
    plt.savefig('roc_pr_lab3_curve.png')

def main():
    X, y = get_X_y()
    X_missing = MCAR_data(X)
    imputers = {
        'Mean': SimpleImputer(strategy='mean'),
        'KNN': KNNImputer(n_neighbors=5),
        'MICE': IterativeImputer(random_state=42)
    }
    imputers_analyze_model(imputers, X_missing, y)

if __name__ == '__main__':
    main()