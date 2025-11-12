from sklearn.experimental import enable_iterative_imputer
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve


data = load_diabetes()
X, y = data.data, data.target

prog = (y > np.median(y)).astype(int)



def simulate_voids_by_percentage(percentage):
    rng = np.random.RandomState(42)
    missing_mask = rng.rand(*X.shape) < percentage  # 10% braków
    X_missing = X.copy()
    X_missing[missing_mask] = np.nan
    return X_missing



imputers = {
    'Mean': SimpleImputer(strategy='mean'),
    'KNN': KNNImputer(n_neighbors=5),
    'MICE': IterativeImputer(random_state=42)
}


def test_imputers(imputers, X_missing, y, output_frame='charts.png'):
    plt.figure(figsize=(12, 5))

    for name, imputer in imputers.items():
        X_imputed = imputer.fit_transform(X_missing)

        X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]

        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
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

    plt.tight_layout()
    plt.savefig(output_frame)


def main():
    X_missing = simulate_voids_by_percentage(percentage=0.2)
    test_imputers(imputers, X_missing, prog, output_frame='charts0_2.png')
    X_missing = simulate_voids_by_percentage(percentage=0.05)
    test_imputers(imputers, X_missing, prog, output_frame='charts0_05.png')


if __name__ == "__main__":
    main()
