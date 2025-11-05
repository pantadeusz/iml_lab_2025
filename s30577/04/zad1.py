import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import tensorflow as tf

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_prob = rf.predict_proba(X_test)[:, 1]
rf_pred = (rf_prob >= 0.5).astype(int)
rf_acc = accuracy_score(y_test, rf_pred)
rf_prec = precision_score(y_test, rf_pred)
rf_rec = recall_score(y_test, rf_pred)
rf_auc = roc_auc_score(y_test, rf_prob)
rf_cm = confusion_matrix(y_test, rf_pred)

print(f"RandomForest -> acc:{rf_acc:.4f} prec:{rf_prec:.4f} rec:{rf_rec:.4f} auc:{rf_auc:.4f}")
print("RF Confusion matrix:\n", rf_cm)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train).astype("float32")
X_test_s = scaler.transform(X_test).astype("float32")
y_train_f = y_train.values.astype("float32")
y_test_f = y_test.values.astype("float32")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train_s.shape[1],)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
model.fit(X_train_s, y_train_f, validation_split=0.2, epochs=50, batch_size=32, verbose=0)

dnn_prob = model.predict(X_test_s, verbose=0).ravel()
dnn_pred = (dnn_prob >= 0.5).astype(int)
dnn_acc = accuracy_score(y_test, dnn_pred)
dnn_prec = precision_score(y_test, dnn_pred)
dnn_rec = recall_score(y_test, dnn_pred)
dnn_auc = roc_auc_score(y_test, dnn_prob)
dnn_cm = confusion_matrix(y_test, dnn_pred)

print(f"DNN (Keras) -> acc:{dnn_acc:.4f} prec:{dnn_prec:.4f} rec:{dnn_rec:.4f} auc:{dnn_auc:.4f}")
print("DNN Confusion matrix:\n", dnn_cm)
