from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt

# Załaduj dane
data = load_diabetes()
X, y = data.data, data.target


# zmiana symulacji brakow na funkcje dla lepszej czytelności, wywolywanie z roznymi parametrami
def make_missing(X, missing_rate=0.1, seed=42):
    rng = np.random.RandomState(seed)
    missing_mask = rng.rand(*X.shape) < missing_rate
    X_missing = X.copy()
    X_missing[missing_mask] = np.nan
    return X_missing


# Metody imputacji
imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "KNN": KNNImputer(n_neighbors=5),
    "MICE": IterativeImputer(random_state=42, max_iter=20),
}

# modele do porowania
models = {
    "Linear": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(random_state=42),
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
}


# 4 funkcja do ewaluacji modelu
def evaluate_model(X_missing, y, imputer, model):

    # podzial na train/test przed imputacja
    X_train, X_test, y_train, y_test = train_test_split(
        X_missing, y, test_size=0.2, random_state=42
    )

    # imputacja fit-transform na zbiorze treningowym i transform na testowym
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    # trenowanie modelu
    model.fit(X_train_imp, y_train)
    y_pred = model.predict(X_test_imp)

    # metryki
    return {
        "R2": r2_score(y_test, y_pred),
    }


# tabela zbiorcza do gromadzenia wynikow
results = []

missing_rates = [0.05, 0.1, 0.2]

# glowna petla eksperymentu
for missing_rate in missing_rates:
    X_missing = make_missing(X, missing_rate)
    for imp_name, imputer in imputers.items():
        for model_name, model in models.items():
            metrics = evaluate_model(X_missing, y, imputer, model)
            results.append(
                {
                    "Missing %": missing_rate * 100,
                    "Imputer": imp_name,
                    "Model": model_name,
                    **metrics,
                }
            )


df = pd.DataFrame(results)
print(df)


# wykres porownania metryki R2 dla roznych modeli i metod imputacji
plt.figure(figsize=(10, 6))
for model_name in models.keys():
    subset = df[df["Model"] == model_name]
    for imp_name in imputers.keys():
        imp_data = subset[subset["Imputer"] == imp_name]
        plt.plot(
            imp_data["Missing %"],
            imp_data["R2"],
            marker="o",
            label=f"{model_name}-{imp_name}",
        )

plt.title("Porównanie R² dla różnych modeli i metod imputacji")
plt.xlabel("Procent braków danych")
plt.ylabel("R² score")
plt.legend()
plt.grid(True)
plt.savefig("wyniki_R2.png")
plt.close()
