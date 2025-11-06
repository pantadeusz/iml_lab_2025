from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from sklearn.utils import class_weight
import numpy as np
import pandas as pd
import time
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

RANDOM_SEED = 50


def prepare_data():
    data = fetch_openml(data_id=1590, as_frame=True, parser="auto")
    X, y = data.data.copy(), data.target.copy()
    X = X.replace("?", pd.NA)
    y = y.map({"<=50K": 0, ">50K": 1}).astype(int)

    num_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols = X.select_dtypes(exclude=[np.number]).columns

    num_pipeline = Pipeline(
        [
            ("imputer", IterativeImputer(random_state=RANDOM_SEED, max_iter=10)),
            ("scaler", StandardScaler()),
        ]
    )

    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [("num", num_pipeline, num_cols), ("cat", cat_pipeline, cat_cols)]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    X_train_prepared = preprocessor.fit_transform(X_train)
    X_test_prepared = preprocessor.transform(X_test)

    try:
        from scipy import sparse

        if sparse.issparse(X_train_prepared):
            X_train_prepared = X_train_prepared.toarray()
        if sparse.issparse(X_test_prepared):
            X_test_prepared = X_test_prepared.toarray()
    except Exception:
        pass

    return X_train_prepared, X_test_prepared, y_train.to_numpy(), y_test.to_numpy()


def build_rf_and_evaluate(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(
        n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1, class_weight="balanced"
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("=== Random Forest ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4))


def build_dnn_and_evaluate(X_train, X_test, y_train, y_test):
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    X_train_tune, X_val, y_train_tune, y_val = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train
    )

    input_shape = (X_train.shape[1],)

    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Input(shape=input_shape))
        model.add(
            layers.Dense(
                units=hp.Int("units_1", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )
        model.add(layers.Dropout(hp.Float("dropout_1", 0.1, 0.5, step=0.1)))
        model.add(
            layers.Dense(
                units=hp.Int("units_2", min_value=16, max_value=256, step=16),
                activation="relu",
            )
        )
        model.add(layers.Dropout(hp.Float("dropout_2", 0.1, 0.5, step=0.1)))
        model.add(layers.Dense(1, activation="sigmoid"))
        hp_lr = hp.Choice("learning_rate", values=[1e-3, 5e-4, 1e-4])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_lr),
            loss="binary_crossentropy",
            metrics=[
                keras.metrics.BinaryAccuracy(name="accuracy"),
                keras.metrics.AUC(name="auc"),
            ],
        )
        return model

    tuner = kt.RandomSearch(
        build_model,
        objective="val_auc",
        max_trials=20,
        executions_per_trial=1,
        directory="keras_tuner_dir",
        project_name="adult_dnn_tuning_improved",
    )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=6, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    tuner.search(
        X_train_tune,
        y_train_tune,
        epochs=20,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        batch_size=256,
    )

    tuner.results_summary(1)

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(best_hps.values)

    y_pred_probs = best_model.predict(X_test).ravel()
    y_pred = (y_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_probs)
    cm = confusion_matrix(y_test, y_pred)

    print("=== DNN (Keras Tuner) ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1:        {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    print("Confusion matrix:")
    print(cm)
    print("\nClassification report:\n")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))


def main():
    X_train, X_test, y_train, y_test = prepare_data()
    build_dnn_and_evaluate(X_train, X_test, y_train, y_test)
    build_rf_and_evaluate(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()
