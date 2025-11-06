import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#autotuner
import keras_tuner as kt

# seed
np.random.seed(42)
tf.random.set_seed(42)

#dane
def load_data():
    X, y = load_breast_cancer(return_X_y=True)
    # 60/20/20: train/val/test
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=42)
    # skalowanie dla DNN
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_va_s = sc.transform(X_va)
    X_te_s = sc.transform(X_te)
    return (X_tr_s, y_tr), (X_va_s, y_va), (X_te_s, y_te)

#baseline (sklearn)
def train_eval_sklearn(X_tr, y_tr, X_va, y_va, X_te, y_te):
    pipe = Pipeline([("scaler", StandardScaler(with_mean=False)),  # dane już są zeskalowane; to "no-op"
                     ("clf", LogisticRegression(max_iter=5000, random_state=42))])
    pipe.fit(X_tr, y_tr)
    # walidacja
    y_pred_va = pipe.predict(X_va)
    print("\n[SKLEARN] walidacja")
    print(confusion_matrix(y_va, y_pred_va))
    print(classification_report(y_va, y_pred_va, digits=3))
    # test
    y_pred_te = pipe.predict(X_te)
    print("[SKLEARN] test")
    print(confusion_matrix(y_te, y_pred_te))
    print(classification_report(y_te, y_pred_te, digits=3))
    return pipe

#DNN (Keras)
def build_dnn(input_dim, units1=64, units2=32, lr=1e-3, dropout=0.0):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(units1, activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(units2, activation="relu"),
        layers.Dense(1, activation="sigmoid"),  # binarnie
    ])
    model.compile(optimizer=keras.optimizers.Adam(lr),
                  loss="binary_crossentropy",
                  metrics=["accuracy", keras.metrics.AUC(name="auc")])
    return model

def train_eval_dnn(X_tr, y_tr, X_va, y_va, X_te, y_te, epochs=30, batch_size=32):
    model = build_dnn(X_tr.shape[1])
    es = keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True, monitor="val_auc", mode="max")
    model.fit(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=batch_size, verbose=0, callbacks=[es])
    # walidacja
    y_pred_va = (model.predict(X_va, verbose=0).ravel() >= 0.5).astype(int)
    print("\n[DNN] walidacja")
    print(confusion_matrix(y_va, y_pred_va))
    print(classification_report(y_va, y_pred_va, digits=3))
    # test
    y_pred_te = (model.predict(X_te, verbose=0).ravel() >= 0.5).astype(int)
    print("[DNN] test")
    print(confusion_matrix(y_te, y_pred_te))
    print(classification_report(y_te, y_pred_te, digits=3))
    return model

#Keras Tuner
def build_dnn_hp(hp: kt.HyperParameters):
    units1 = hp.Int("units1", 32, 256, step=32)
    units2 = hp.Int("units2", 16, 128, step=16)
    dropout = hp.Choice("dropout", [0.0, 0.2, 0.4])
    lr = hp.Choice("lr", [1e-3, 3e-4, 1e-4])
    return build_dnn(input_dim=hp.Fixed("input_dim", 30), units1=units1, units2=units2, lr=lr, dropout=dropout)

def tune_dnn(X_tr, y_tr, X_va, y_va, max_trials=10, epochs=25):
    hp_input = X_tr.shape[1]
    def _wrapped(hp):
        hp.Fixed("input_dim", hp_input)
        return build_dnn_hp(hp)
    tuner = kt.RandomSearch(
        _wrapped, objective=kt.Objective("val_auc", direction="max"),
        max_trials=max_trials, overwrite=True, directory="kt_dir", project_name="lab5"
    )
    es = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_auc", mode="max")
    tuner.search(X_tr, y_tr, validation_data=(X_va, y_va), epochs=epochs, batch_size=32, verbose=0, callbacks=[es])
    best = tuner.get_best_models(1)[0]
    return best

if __name__ == "__main__":
    (X_tr, y_tr), (X_va, y_va), (X_te, y_te) = load_data()

    # 1) baseline sklearn
    _ = train_eval_sklearn(X_tr, y_tr, X_va, y_va, X_te, y_te)

    # 2) baseline DNN
    _ = train_eval_dnn(X_tr, y_tr, X_va, y_va, X_te, y_te, epochs=30)

    # 3) tuner
    best_model = tune_dnn(X_tr, y_tr, X_va, y_va, max_trials=8, epochs=20)
    # ocena modelu z tunera
    y_pred_va = (best_model.predict(X_va, verbose=0).ravel() >= 0.5).astype(int)
    print("\n[DNN tuned] walidacja")
    print(confusion_matrix(y_va, y_pred_va))
    print(classification_report(y_va, y_pred_va, digits=3))
    y_pred_te = (best_model.predict(X_te, verbose=0).ravel() >= 0.5).astype(int)
    print("[DNN tuned] test")
    print(confusion_matrix(y_te, y_pred_te))
    print(classification_report(y_te, y_pred_te, digits=3))