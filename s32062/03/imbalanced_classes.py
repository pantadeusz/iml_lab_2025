
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

threshold = 0.1

# Generuj dane niezbalansowane
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bazowy model
model = LogisticRegression(random_state=42)
# ... trenuj i oceń
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, output_dict=True))
# Z ważeniem klas
model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
# ...
model_weighted.fit(X_train, y_train)
y_pred_weighted = (model_weighted.predict_proba(X_test)[:,1] >= threshold).astype(int)
print(classification_report(y_test, y_pred_weighted, output_dict=True))

smote = SMOTE(sampling_strategy='auto', random_state=42)
X_smote, y_smote = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_smote, y_smote)
y_pred_smote = (model_smote.predict_proba(X_test)[:,1] >= threshold).astype(int)
print(classification_report(y_test, y_pred_smote, output_dict=True))

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
model_rus = LogisticRegression(random_state=42)
model_rus.fit(X_train_rus, y_train_rus)
y_pred_rus = (model_rus.predict_proba(X_test)[:,1] >= threshold).astype(int)
print(classification_report(y_test, y_pred_rus, output_dict=True))


