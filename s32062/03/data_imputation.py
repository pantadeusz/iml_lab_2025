import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import classification_report, mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# Załaduj dane
data = load_diabetes()
X, y = data.data, data.target
# Symuluj braki (MCAR)
rng = np.random.RandomState(42)
missing_mask = rng.rand(*X.shape) < 0.22  # 10% braków
X_missing = X.copy()
X_missing[missing_mask] = np.nan

# Metody imputacji
imputers = {
    'Mean': SimpleImputer(strategy='mean'),
    'KNN': KNNImputer(n_neighbors=5),
    'MICE': IterativeImputer(random_state=42)
}

model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Model without imputed data:")
print(f"MSE: {mean_squared_error(y_test, y_pred)}")
print(f"  R² Score: {r2_score(y_test, y_pred):.4f}")
print(f"  MAE: {mean_absolute_error(y_test, y_pred):.2f}")
print("Models with imputed data:")
for name, imputer in imputers.items():
    X_imputed = imputer.fit_transform(X_missing)
    # # Trenuj model i oceń
    model_with_imputed = LinearRegression()
    # ... (podziel na train/test, trenuj, oceń)
    X_train_imputed, X_test_imputed, y_train_for_imputed, y_test_for_imputed = train_test_split(X_imputed, y, test_size = 0.2, random_state=42)
    model_with_imputed.fit(X_train_imputed, y_train_for_imputed)
    y_pred_with_imputed = model_with_imputed.predict(X_test_imputed)
    print(f'==={name} imputation===')

    print(f"MSE: {mean_squared_error(y_test_for_imputed, y_pred_with_imputed)}")
    print(f"  R² Score: {r2_score(y_test_for_imputed, y_pred_with_imputed):.4f}")
    print(f"  MAE: {mean_absolute_error(y_test_for_imputed, y_pred_with_imputed):.2f}")



