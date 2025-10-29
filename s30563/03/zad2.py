from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def predict(model, X):
    threshold = 0.5
    return model.predict_proba(X)[:, 1] >= threshold

# Generuj dane niezbalansowane
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, weights=[0.95, 0.05], random_state=42)

# Bazowy model
X_train,X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred_base = predict(model, X_test)

print("Model bazowy:")
print(classification_report(y_test, y_pred_base, output_dict=True))

# Z ważeniem klas
model_weighted = LogisticRegression(class_weight='balanced', random_state=42)
model_weighted.fit(X_train, y_train)

y_pred_weighted = predict(model_weighted, X_test)

print("Model z ważeniem klas:")
print(classification_report(y_test, y_pred_weighted, output_dict=True))

# oversampling
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

model_smote = LogisticRegression(random_state=42)
model_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = predict(model_smote, X_test)
print("Model SMOTE:")
print(classification_report(y_test, y_pred_smote, output_dict=True))

# undersampling
rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

model_rus = LogisticRegression(random_state=42)
model_rus.fit(X_train_rus, y_train_rus)

y_pred_rus = predict(model_rus, X_test)
print("Model RUS:")
print(classification_report(y_test, y_pred_rus, output_dict=True))
