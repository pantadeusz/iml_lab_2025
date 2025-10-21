from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler


# Generuj dane niezbalansowane
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train, y_train)

# Bazowy model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba >= 0.5).astype(int)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f'Base: acc={acc:.3f}, prec={prec:.3f}, rec={rec:.3f}, f1={f1:.3f}')

# Z ważeniem klas
model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)
y_proba_weighted = model_weighted.predict_proba(X_test)[:, 1]
y_pred_weighted = (y_proba_weighted >= 0.5).astype(int)
acc_w = accuracy_score(y_test, y_pred_weighted)
prec_w = precision_score(y_test, y_pred_weighted, zero_division=0)
rec_w = recall_score(y_test, y_pred_weighted)
f1_w = f1_score(y_test, y_pred_weighted)
print(f'Weighted: acc={acc_w:.3f}, prec={prec_w:.3f}, rec={rec_w:.3f}, f1={f1_w:.3f}')

# ważenie klas z progiem
threshold = 0.3
y_pred_weighted_thr = (y_proba_weighted >= threshold).astype(int)
acc_w_thr = accuracy_score(y_test, y_pred_weighted_thr)
prec_w_thr = precision_score(y_test, y_pred_weighted_thr, zero_division=0)
rec_w_thr = recall_score(y_test, y_pred_weighted_thr)
f1_w_thr = f1_score(y_test, y_pred_weighted_thr)
print(f'Weighted (thr=0.3): acc={acc_w_thr:.3f}, prec={prec_w_thr:.3f}, rec={rec_w_thr:.3f}, f1={f1_w_thr:.3f}')

# SMOTE
model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
y_proba_smote = model_smote.predict_proba(X_test)[:, 1]
y_pred_smote = (y_proba_smote >= 0.5).astype(int)
acc_s = accuracy_score(y_test, y_pred_smote)
prec_s = precision_score(y_test, y_pred_smote, zero_division=0)
rec_s = recall_score(y_test, y_pred_smote)
f1_s = f1_score(y_test, y_pred_smote)
print(f'SMOTE: acc={acc_s:.3f}, prec={prec_s:.3f}, rec={rec_s:.3f}, f1={f1_s:.3f}')

# Undersampling
model_under = LogisticRegression(random_state=42)
model_under.fit(X_train_under, y_train_under)
y_proba_under = model_under.predict_proba(X_test)[:, 1]
y_pred_under = (y_proba_under >= 0.5).astype(int)
acc_u = accuracy_score(y_test, y_pred_under)
prec_u = precision_score(y_test, y_pred_under, zero_division=0)
rec_u = recall_score(y_test, y_pred_under)
f1_u = f1_score(y_test, y_pred_under)
print(f'Undersample: acc={acc_u:.3f}, prec={prec_u:.3f}, rec={rec_u:.3f}, f1={f1_u:.3f}')