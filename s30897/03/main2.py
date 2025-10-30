import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Generowanie danych
X, y = make_classification(n_samples=1000,n_features=20,n_classes=2,weights=[0.95, 0.05], random_state=42)

print(f"Początkowy rozkład klas: {np.bincount(y)}")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y) # 70% na train i 30% na test


# Definiujemy nazwy klas
target_names = ['Klasa 0 (Większość)', 'Klasa 1 (Mniejszość)']

# --- Model Bazowy ---
print("Model Bazowy (bez zmian)")
model_base = LogisticRegression(random_state=42)
model_base.fit(X_train, y_train)
y_pred_base = model_base.predict(X_test)

print("Wyniki Modelu Bazowego")
print(classification_report(y_test, y_pred_base, target_names=target_names, zero_division=0))


# --- Ważenie Klas ---
print("Model z ważeniem klas (class_weight='balanced')")
model_weight = LogisticRegression(class_weight='balanced', random_state=42)
model_weight.fit(X_train, y_train)
y_pred_weight = model_weight.predict(X_test)

print("Wyniki Modelu z Ważeniem Klas")
print(classification_report(y_test, y_pred_weight, target_names=target_names, zero_division=0))


# ---Oversampling (SMOTE) ---
print("Model z Oversamplingiem (SMOTE)")
# SMOTE stosujemy TYLKO na danych treningowych
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Rozkład klas po SMOTE: {np.bincount(y_train_smote)}")

model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = model_smote.predict(X_test)

print("Wyniki Modelu SMOTE")
print(classification_report(y_test, y_pred_smote, target_names=target_names, zero_division=0))


# --- Undersampling (RandomUnderSampler) ---
print("Model z Undersamplingiem (RandomUnderSampler)")
# Undersampling też stosujemy TYLKO na danych treningowych
rand = RandomUnderSampler(random_state=42)
X_train_rand, y_train_rand = rand.fit_resample(X_train, y_train)

print(f"Rozkład klas po RUS: {np.bincount(y_train_rand)}")

model_rand = LogisticRegression(random_state=42)
model_rand.fit(X_train_rand, y_train_rand)
y_pred_rand = model_rand.predict(X_test)

print("Wyniki Modelu RandomUnderSampler")
print(classification_report(y_test, y_pred_rand, target_names=target_names, zero_division=0))