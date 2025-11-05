import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

rng = np.random.RandomState(42)  # seed

# dane i podział
X, y = load_diabetes(return_X_y=True)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

# maski braków
def mask_mcar(shape, rate):  # braki losowo
    return rng.rand(*shape) < rate

def mask_mar(X, rate):  # braki zależne od innej cechy
    n, d = X.shape
    m = np.zeros((n, d), bool)
    driver = np.abs(X[:, 0])
    thr = np.quantile(driver, 1 - rate)
    rows = driver > thr
    cols = np.arange(d)[::2]  # co druga kolumna ma braki
    m[np.ix_(rows, cols)] = True
    return m

def mask_mnar(X, rate):  # braki zależne od wartości tej samej cechy
    n, d = X.shape
    m = np.zeros((n, d), bool)
    for j in range(d):
        col = np.abs(X[:, j])
        thr = np.quantile(col, 1 - rate)
        m[:, j] = col > thr
    return m

# prosta ewaluacja
def eval_reg(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))  # rmse = sqrt(mse)
    return r2, mae, rmse

# baseline: bez braków i bez imputacji (tylko scaler + ridge)
base_pipe = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
base_pipe.fit(X_tr, y_tr)
base_pred = base_pipe.predict(X_te)
base_r2, base_mae, base_rmse = eval_reg(y_te, base_pred)

# scenariusze i imputery
scenarios = [("MCAR", mask_mcar), ("MAR", mask_mar), ("MNAR", mask_mnar)]
rates = [0.05, 0.20]  # 5% i 20% braków
imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "KNN": KNNImputer(n_neighbors=5),
    "MICE": IterativeImputer(random_state=42, max_iter=20, tol=1e-3),
}

rows = []
# pętla: scenariusz -> poziom braków -> imputer -> pipeline
for scen_name, scen_fn in scenarios:
    for rate in rates:
        # generuj braki osobno dla train/test
        if scen_name == "MCAR":
            m_tr = scen_fn(X_tr.shape, rate)
            m_te = scen_fn(X_te.shape, rate)
        else:
            m_tr = scen_fn(X_tr, rate)
            m_te = scen_fn(X_te, rate)

        # zastosuj braki
        X_tr_m = X_tr.copy(); X_tr_m[m_tr] = np.nan
        X_te_m = X_te.copy(); X_te_m[m_te] = np.nan

        # trenowanie i ocena
        for imp_name, imp in imputers.items():
            pipe = Pipeline([
                ("imputer", imp),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ])
            pipe.fit(X_tr_m, y_tr)
            pred = pipe.predict(X_te_m)
            r2, mae, rmse = eval_reg(y_te, pred)
            rows.append({
                "scenario": scen_name, "rate": rate, "imputer": imp_name,
                "R2": r2, "MAE": mae, "RMSE": rmse
            })

# wyniki do tabeli + krótki wydruk
df = pd.DataFrame(rows).sort_values(["scenario", "rate", "imputer"])
print("\nbaseline (no missing, no imputation):")
print(f"R2={base_r2:.4f} MAE={base_mae:.4f} RMSE={base_rmse:.4f}")

print("\nresults (first 12 rows):")
print(df.head(12).to_string(index=False))