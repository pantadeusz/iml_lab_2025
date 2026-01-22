from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os

RANDOM_STATE = 75

data = load_wine()
X, y = data.data, data.target

X_train, X_val, y_train, y_val = train_test_split(
	X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_val)
rf_acc = accuracy_score(y_val, rf_pred)
print(f"RF acc: {rf_acc:.4f}")

es = EarlyStopping(monitor="val_accuracy", mode="max", patience=30, restore_best_weights=True)
norm = layers.Normalization()
norm.adapt(X_train)
model_norm = tf.keras.Sequential([
	layers.Input(shape=(X_train.shape[1],)),
	norm,
	layers.Dense(8, activation="relu"),
	layers.Dense(4, activation="relu"),
	layers.Dense(3, activation="softmax"),
])
model_norm.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model_norm.fit(X_train, y_train, epochs=200, batch_size=16, verbose=0, validation_data=(X_val, y_val), callbacks=[es])
nn_norm_acc = model_norm.evaluate(X_val, y_val, verbose=0)[1]
print(f"NN acc: {nn_norm_acc:.4f}")


def _next_keras_idx():
	i = 1
	while os.path.exists(f"{i}.keras"):
		i += 1
	return i

idx = _next_keras_idx()
out_path = f"{idx}.keras"
model_norm.save(out_path)
print(f"Saved: {out_path} | size={os.path.getsize(out_path)} B")