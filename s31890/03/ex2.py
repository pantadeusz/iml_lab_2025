from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import IterativeImputer
import numpy as np
import matplotlib.pyplot as plt


# Załaduj dane
data = load_diabetes()
if data is None:
    raise FileNotFoundError("Could not find the dataset")

X, y = data.data, data.target

# Symuluj braki (MCAR)
rng = np.random.RandomState(42)
missing = 0.5
missing_mask = rng.rand(*X.shape) < missing  # 10% braków
X_missing = X.copy()
X_missing[missing_mask] = np.nan

# Metody imputacji
imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "KNN": KNNImputer(n_neighbors=5),
    "MICE": IterativeImputer(random_state=42),
}

for name, imputer in imputers.items():
    print(f"\n=== Imputation: {name} || Missing {missing*100}% ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X_missing, y, test_size=0.2, random_state=42
    )

    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Ewaluacja
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R²:   {r2:.3f}")

    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.6, label='Predictions')

    # Ideal
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Ideal Fit (y=x)')

    # Trend
    coeffs = np.polyfit(y_test, y_pred, 1)
    trend = np.poly1d(coeffs)
    y_sorted = np.sort(y_test)
    plt.plot(y_sorted, trend(y_sorted), color='orange', label='Trend line')

    plt.fill_between(y_sorted, y_sorted, trend(y_sorted), color='orange', alpha=0.2, label='Deviation area')

    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.title(f"True vs Predicted with Trend - {name} Imputation. Missing {missing*100}% data")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"true_vs_pred_trend_{name}_{missing*100}%_missing.png")
    plt.close()
