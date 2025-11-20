from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# Załaduj dane
data = load_diabetes()
X, y = data.data, data.target


# Symuluj braki (MCAR)
def simulate_missing(X, missing_rate):
    rng = np.random.RandomState(42)
    missing_mask = rng.rand(*X.shape) < missing_rate
    X_missing = X.copy()
    X_missing[missing_mask] = np.nan
    return X_missing


# Model bez braków
print("MODEL BAZOWY (bez braków)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

base_mse = mean_squared_error(y_test, y_pred)

print(f"MSE: {base_mse:.2f}")
print()

# Metody imputacji
imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "KNN": KNNImputer(n_neighbors=5),
    "MICE": IterativeImputer(random_state=42, max_iter=152),
}

missing_rates = [0.05, 0.10, 0.20, 0.30]
results = []

for missing_rate in missing_rates:
    print(f"BRAKI: {missing_rate * 100:.0f}%")

    X_missing = simulate_missing(X, missing_rate)
    n_missing = np.isnan(X_missing).sum()

    for name, imputer in imputers.items():
        X_imputed = imputer.fit_transform(X_missing)

        X_train, X_test, y_train, y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42
        )

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)

        mse_diff = mse - base_mse

        results.append(
            {
                "Missing_Rate": f"{missing_rate * 100:.0f}%",
                "Method": name,
                "MSE": mse,
                "MSE_diff": mse_diff,
            }
        )

        print(
            f"{name:6} - MSE: {mse:7.2f} (+{mse_diff:6.2f})"
        )

    print()

print("PODSUMOWANIE WSZYSTKICH WYNIKÓW")
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
