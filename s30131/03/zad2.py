import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

RNG = 42

# generuj niezbalansowane dane (95% klasa 0, 5% klasa 1)
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=6, n_redundant=2,
    n_classes=2, weights=[0.95, 0.05], class_sep=1.0, flip_y=0.01,
    random_state=RNG
)

# podział: 60% train, 20% val, 20% test (stratyfikowany)
X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RNG)
X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=RNG)

# strojenie progu pod F1 na walidacji
def tune_threshold_f1(y_true, y_scores):
    p, r, thr = precision_recall_curve(y_true, y_scores)
    thr = np.r_[0.0, thr]  # wyrównaj długość
    f1 = 2 * p * r / (p + r + 1e-12)
    i = np.nanargmax(f1)
    return float(thr[i])

# ewaluacja na teście dla danego progu
def eval_at_threshold(y_true, y_scores, thr):
    y_pred = (y_scores >= thr).astype(int)
    return dict(
        acc=accuracy_score(y_true, y_pred),
        prec=precision_score(y_true, y_pred, zero_division=0),
        rec=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=roc_auc_score(y_true, y_scores),
        ap=average_precision_score(y_true, y_scores),
        thr=thr,
    )

# baseline: scaler + logreg (próg strojony na walidacji)
pipe_base = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=5000, random_state=RNG))])
pipe_base.fit(X_tr, y_tr)
thr_base = tune_threshold_f1(y_val, pipe_base.predict_proba(X_val)[:, 1])
res_base = eval_at_threshold(y_te, pipe_base.predict_proba(X_te)[:, 1], thr_base)

# class_weight='balanced'
pipe_w = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(class_weight="balanced", max_iter=5000, random_state=RNG))])
pipe_w.fit(X_tr, y_tr)
thr_w = tune_threshold_f1(y_val, pipe_w.predict_proba(X_val)[:, 1])
res_w = eval_at_threshold(y_te, pipe_w.predict_proba(X_te)[:, 1], thr_w)

# SMOTE oversampling (sampluj tylko na train)
pipe_smote = ImbPipeline([
    ("smote", SMOTE(random_state=RNG)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, random_state=RNG)),
])
pipe_smote.fit(X_tr, y_tr)
thr_smote = tune_threshold_f1(y_val, pipe_smote.predict_proba(X_val)[:, 1])
res_smote = eval_at_threshold(y_te, pipe_smote.predict_proba(X_te)[:, 1], thr_smote)

# undersampling
pipe_under = ImbPipeline([
    ("under", RandomUnderSampler(random_state=RNG)),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=5000, random_state=RNG)),
])
pipe_under.fit(X_tr, y_tr)
thr_under = tune_threshold_f1(y_val, pipe_under.predict_proba(X_val)[:, 1])
res_under = eval_at_threshold(y_te, pipe_under.predict_proba(X_te)[:, 1], thr_under)

# dodatkowo: baseline przy progu 0.5
res_base05 = eval_at_threshold(y_te, pipe_base.predict_proba(X_te)[:, 1], 0.5)

# wydruk krótkich wyników
def p(name, r):
    print(f"{name:12s} thr={r['thr']:.3f} acc={r['acc']:.3f} prec={r['prec']:.3f} rec={r['rec']:.3f} f1={r['f1']:.3f} roc_auc={r['roc_auc']:.3f} ap={r['ap']:.3f}")

print("\n=== wyniki (próg strojony pod F1 na walidacji) ===")
p("baseline", res_base)
p("class_weight", res_w)
p("smote", res_smote)
p("undersample", res_under)

print("\n--- baseline z progiem 0.5 (odniesienie) ---")
p("base@0.5", res_base05)