import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

# Załaduj dane
data = load_diabetes()
X, y = data.data, data.target

# Symuluj braki (MCAR)
rng = np.random.RandomState(42)
missing_mask = rng.rand(*X.shape) < 0.2  # 10% braków
X_missing = X.copy()
X_missing[missing_mask] = np.nan

# Metody imputacji
imputers = {
    'Mean': SimpleImputer(strategy='mean'),
    'KNN': KNNImputer(n_neighbors=5),
    'MICE': IterativeImputer(random_state=42)
}

models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=100)
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_miss, X_test_miss = train_test_split(X_missing, test_size=0.2, random_state=42)

base_model = LinearRegression()
base_model.fit(X_train, y_train)
y_pred = base_model.predict(X_test)

base_r2 = r2_score(y_test, y_pred)
base_rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("=== Model bazowy (bez braków) ===")
print(f"R²: {base_r2:.3f}")
print(f"RMSE: {base_rmse:.3f}\n")



# ============= ROZSZERZENIE: Porównanie z różnymi modelami =============

print("\n" + "=" * 70)
print("PORÓWNANIE IMPUTACJI Z RÓŻNYMI TYPAMI MODELI")
print("=" * 70 + "\n")

results = []

for imputer_name, imputer in imputers.items():
    X_imputed = imputer.fit_transform(X_train_miss)
    X_test_imp = imputer.transform(X_test_miss)

    for model_name in models.items():
        if model_name == 'LinearRegression':
            model = LinearRegression()
        elif model_name == 'DecisionTree':
            model = DecisionTreeRegressor(random_state=42)
        else:
            model = RandomForestRegressor(random_state=42, n_estimators=100)

        model.fit(X_imputed, y_train)
        y_pred_imp = model.predict(X_test_imp)

        r2 = r2_score(y_test, y_pred_imp)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_imp))

        results.append({
            'Imputacja': imputer_name,
            'Model': model_name,
            'R²': r2,
            'RMSE': rmse
        })

        print(f"{imputer_name:15} + {model_name:18} | R²: {r2:.4f} | RMSE: {rmse:.4f}")