from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = load_diabetes()
X, y = data.data, data.target
feature_names = data.feature_names

rng = np.random.RandomState(42)
missing_mask = rng.rand(*X.shape) < 0.1  # 10% missing
X_missing = X.copy()
X_missing[missing_mask] = np.nan

imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "KNN": KNNImputer(n_neighbors=5),
    "MICE": IterativeImputer(random_state=42),
}

models = [
    ("Linear Regression", LinearRegression()),
    ("Random Forest", RandomForestRegressor(random_state=42)),
    ("XGBoost", XGBRegressor(random_state=42)),
]


def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    return mse


X_train_miss, X_test_miss, y_train, y_test = train_test_split(
    X_missing, y, test_size=0.2, random_state=42
)

X_train_miss_df = pd.DataFrame(X_train_miss, columns=feature_names)
X_test_miss_df = pd.DataFrame(X_test_miss, columns=feature_names)

X_train_drop = X_train_miss_df.dropna(axis=0, how="any")
y_train_drop = y_train[X_train_drop.index]
X_test_drop = X_test_miss_df.dropna(axis=0, how="any")
y_test_drop = y_test[X_test_drop.index]

stats = []

for model_name, model in models:
    mse = evaluate_model(
        model, X_train_drop.values, X_test_drop.values, y_train_drop, y_test_drop
    )
    stats.append({"Imputer": "DROPPED", "Model": model_name, "MSE": mse})

for model_name, model in models:
    for imputer_name, imputer in imputers.items():
        X_train_imp = imputer.fit_transform(X_train_miss)
        X_test_imp = imputer.transform(X_test_miss) # prevent information leakage without
        mse = evaluate_model(model, X_train_imp, X_test_imp, y_train, y_test)
        stats.append({"Imputer": imputer_name, "Model": model_name, "MSE": mse})

df_stats = pd.DataFrame(stats)
df_stats = df_stats.sort_values(by="MSE")

plt.figure(figsize=(10, 6))
sns.barplot(data=df_stats, x="Imputer", y="MSE", hue="Model", palette="rocket")
plt.title("Comparison of Imputation Methods", fontsize=14)
plt.xlabel("Imputation Method")
plt.ylabel("Mean Squared Error (MSE)")
plt.legend(title="Model", loc="upper right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
