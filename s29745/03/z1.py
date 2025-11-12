from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
    "MICE": IterativeImputer(random_state=42, max_iter=1000)
}

results = {}

# Trenowanie modelu bez brakow
model = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
results["Non imputed"] = mse

# Trenowanie modeli z imputerem
for name, imputer in imputers.items():
    X_imputed = imputer.fit_transform(X_missing)
    # Trenuj model i oceń
    # ... (podziel na train/test, trenuj, oceń)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    results[name] = mse

for method, mse in results.items():
    print(f"{method}: MSE = {mse:.2f}")
