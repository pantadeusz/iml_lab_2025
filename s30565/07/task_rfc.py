from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import os
from joblib import dump


wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

base_model = RandomForestClassifier(random_state=42)
base_model.fit(X_train, y_train)
y_base_pred = base_model.predict(X_test)
print(classification_report(y_test, y_base_pred))

def sizeof_fmt(num, suffix='B'):
    for unit in ['','K','M','G','T','P']:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} P{suffix}"

os.makedirs('models', exist_ok=True)

pipeline = make_pipeline(scaler, base_model)
dump(pipeline, 'models/rf_pipeline.joblib')

path = 'models/rf_pipeline.joblib'
size = os.path.getsize(path)
print(f"Zapisano: `{path}` — rozmiar: {sizeof_fmt(size)}")