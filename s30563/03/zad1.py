import os

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# Załaduj dane
data = load_diabetes()
X, y = data.data, data.target

# Symuluj braki (MCAR)
rng = np.random.RandomState(42)
missing_percentage = 0.60
missing_mask = rng.rand(*X.shape) < missing_percentage  # 10% braków
X_missing = X.copy()
X_missing[missing_mask] = np.nan

# Metody imputacji
imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "KNN": KNNImputer(n_neighbors=5),
    "MICE": IterativeImputer(random_state=42),
}

# Model bazowy (bez braków)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

base_model = LinearRegression()
base_model.fit(X_train, y_train)

y_pred = base_model.predict(X_test)

base_r2 = r2_score(y_test, y_pred)
base_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

results = []

results.append({
    "Missing": missing_percentage,
    "Method" : "Base model",
    "R2": base_r2,
    "RMSE": base_rmse
})

X_train_miss, X_test_miss = train_test_split(X_missing, test_size=0.2, random_state=42)

for name, imputer in imputers.items():
    X_imputed = imputer.fit_transform(X_train_miss)
    X_test_imputed = imputer.transform(X_test_miss)
    # Trenuj model i oceń
    model = LinearRegression()
    model.fit(X_imputed, y_train)
    y_pred = model.predict(X_test_imputed)
    imputed_r2 = r2_score(y_test, y_pred)
    imputed_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results.append({
        "Missing": missing_percentage,
        "Method": name,
        "R2": imputed_r2,
        "RMSE": imputed_rmse
    })

df_results = pd.DataFrame(results)
print(df_results)

# file_path = "results.csv"
#
# if not os.path.exists(file_path):
#     df_results.to_csv(file_path, index=False)
# else:
#     df_results[1:].to_csv(file_path, mode='a', header=False, index=False)
