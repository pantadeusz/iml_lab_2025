import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE

# Generuj dane niezbalansowane
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Baza

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Base model")
print(classification_report(y_test, y_pred, output_dict=True))

# Wazenie

model_w = LogisticRegression(class_weight='balanced', random_state=42)
model_w.fit(X_train, y_train)
y_pred_w = model_w.predict(X_test)
print("Ważenie klas")
print(classification_report(y_test, y_pred_w, output_dict=True))

# Oversampling

Xy_train = np.hstack([X_train, y_train.reshape(-1,1)]) # łącze x i y, y reshapuje zeby pasował do x i wtedy tworzy sie jedna tabela
df_train = pd.DataFrame(Xy_train)

df_min = df_train[df_train.iloc[:, -1] == 1] # iloc to indexowanie pozycyjne w df, bierzemy wszystkie wiersze ":" i kolumne -1
# czyli ostatnia i patrzymy czy rowna sie 1, zwraca wszystkie wiersze spelniajace ten warunek
df_maj = df_train[df_train.iloc[:, -1] == 0]

df_min_upsampled = resample(df_min,
                            replace=True, #moga sie powtarzac
                            n_samples=len(df_maj), #tyle samo co wiekszosci
                            random_state=42)

df_balanced_overs = pd.concat([df_maj, df_min_upsampled]) # łączymy

X_train_over = df_balanced_overs.iloc[:, :-1].values #wszystkie kolumny oprocz ostatniej
y_train_over = df_balanced_overs.iloc[:, -1].values.astype(int) #tylko ostatnia kolumna


model_upsampled = LogisticRegression(random_state=42, max_iter=1000)
model_upsampled.fit(X_train_over, y_train_over)
y_pred_upsampled = model_upsampled.predict(X_test)
print("Upsampled model manualnie")
print(classification_report(y_test, y_pred_upsampled, output_dict=True))

# SMOTE

sm = SMOTE(random_state=42)
X_sm, y_sm = sm.fit_resample(X_train, y_train)

model_smote = LogisticRegression(random_state=42, max_iter=1000)
model_smote.fit(X_sm, y_sm)
y_pred_smote = model_smote.predict(X_test)
print("SMOTE model")
print(classification_report(y_test, y_pred_smote, output_dict=True))

#Undersampling

df_maj_down = resample(df_maj,
                       replace=False,
                       n_samples=len(df_min),
                       random_state=42)

df_balanced_under = pd.concat([df_maj_down, df_min])

X_train_under = df_balanced_under.iloc[:, :-1].values
y_train_under = df_balanced_under.iloc[:, -1].values.astype(int)

model_undersampled = LogisticRegression(random_state=42, max_iter=1000)
model_undersampled.fit(X_train_under, y_train_under)
y_pred_under = model_undersampled.predict(X_test)
print("Undersampled model")
print(classification_report(y_test, y_pred_under, output_dict=True))

