from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd

# Załaduj dane
data = load_diabetes()
X, y = data.data, data.target

# Symuluj braki (MCAR)
rng = np.random.RandomState(42)
missing_mask = rng.rand(*X.shape) < 0.1  # 10% braków
X_missing = X.copy()
X_missing[missing_mask] = np.nan

# Metody imputacji
imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "KNN": KNNImputer(n_neighbors=5),
    "MICE": IterativeImputer(random_state=42),
}

for name, imputer in imputers.items():
    X_imputed = imputer.fit_transform(X_missing)
    # Trenuj model i oceń
    model = LogisticRegression(random_state=42)
    # ... (podziel na train/test, trenuj, oceń)
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
