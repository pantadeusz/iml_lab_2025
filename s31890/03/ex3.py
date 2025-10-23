from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

strategies = {
    "Original": None,
    "SMOTE": SMOTE(random_state=42),
    "Undersampling": RandomUnderSampler(random_state=42),
    "Class Weights": "balanced"
}

# Załaduj dane
X, y = make_classification(
    n_samples=1000,
    n_features=20,
    n_informative=2,
    n_redundant=10,
    n_clusters_per_class=1,
    weights=[0.9, 0.1],
    flip_y=0,
    random_state=1
)
# Podziel dane
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1
)

plt.figure(figsize=(10, 6))

for name, sampler in strategies.items():
    X_train_bal, y_train_bal = X_train.copy(), y_train.copy()
    
    if name == "SMOTE":
        X_train_bal, y_train_bal = sampler.fit_resample(X_train, y_train)
    elif name == "Undersampling":
        X_train_bal, y_train_bal = sampler.fit_resample(X_train, y_train)

    if sampler is None or name == "Class Weights":
        model = LogisticRegression(random_state=42, max_iter=1000)
        if name == "Class Weights":
            model = LogisticRegression(random_state=42, max_iter=1000, class_weight="balanced")
        model.fit(X_train_bal, y_train_bal)
        y_proba = model.predict_proba(X_test)[:,1]
    else:
        # SMOTE lub Undersampling
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(random_state=42, max_iter=1000))
        ])
        pipeline.fit(X_train_bal, y_train_bal)
        y_proba = pipeline.predict_proba(X_test)[:,1]

    # PR
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall, precision)
    
    plt.plot(recall, precision, label=f"{name} (AUC = {pr_auc:.4f})")

    print(f"{name}: PR-AUC = {pr_auc:.4f}")
    
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("PR Curves: Imbalance Handling Strategies")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("pr_curves_simple.png", dpi=200)
plt.close()
