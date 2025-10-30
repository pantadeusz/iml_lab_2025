from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

X, y = make_classification(
    n_samples=1000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred)
    }


results = []

model = LogisticRegression(random_state=42)
results.append(evaluate_model("Original", model, X_train, y_train, X_test, y_test))

model_weighted = LogisticRegression(class_weight="balanced", random_state=42)
results.append(evaluate_model("Class Weighted", model_weighted, X_train, y_train, X_test, y_test))

smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression(random_state=42)
results.append(evaluate_model("SMOTE Oversampling", model_smote, X_train_sm, y_train_sm, X_test, y_test))

under = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = under.fit_resample(X_train, y_train)
model_under = LogisticRegression(random_state=42)
results.append(evaluate_model("Undersampling", model_under, X_train_under, y_train_under, X_test, y_test))

df_results = pd.DataFrame(results)
print(df_results)

df_melted = df_results.melt(
    id_vars="Model",
    var_name="Metric",
    value_name="Score"
)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=df_melted,
    x="Model",
    y="Score",
    hue="Metric",
    palette="viridis"
)
plt.title("Comparison of Model Performance by Metric")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(title="Metric", loc="upper right")
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
