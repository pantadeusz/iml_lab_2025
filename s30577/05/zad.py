import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, roc_auc_score, classification_report, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt


def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y

def prepare_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def train_baseline_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test, model_name="Model"):
    
    y_pred = model.predict(X_test)

    if hasattr(model, 'predict_proba'):
        y_pred_classes = y_pred  
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    else:
        y_pred_classes = (y_pred > 0.5).astype(int).flatten()
        y_pred_proba = y_pred.flatten()    

    accuracy = accuracy_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes, average="weighted")
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\n=== {model_name} ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score (weighted): {f1:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes))

def build_dnn_model(input_dim):
    model = keras.Sequential([
        layers.Dense(64, activation="relu", input_shape=(input_dim,)),
        layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model

def train_dnn_model(X_train, y_train, X_test, y_test, epochs=30):
    model = build_dnn_model(X_train.shape[1])
    model.fit(
        X_train, y_train, 
        epochs=epochs, 
        batch_size=32, 
        validation_data=(X_test, y_test),
        verbose=0
    )
    return model

def build_tuner_model(input_dim):
    def build_model(hp):
        model = keras.Sequential()
        model.add(layers.Dense(
            units=hp.Int("units", min_value=32, max_value=128, step=32),
            activation="relu",
            input_shape=(input_dim,)
        ))
        model.add(layers.Dense(1, activation="sigmoid"))
        learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model
    
    return build_model

def run_tuner(X_train, y_train, X_test, y_test, max_trials=6, epochs=30):
    build_model = build_tuner_model(input_dim=X_train.shape[1])
    print(f"Max trials: {max_trials}, Epochs: {epochs}")
    project_name = f"dnn_tuner_t{max_trials}_e{epochs}"

    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=max_trials,
        directory='tuner_results',
        project_name=project_name,
        overwrite=True 
    )
    tuner.search(
        X_train, y_train, 
        epochs=epochs,
        batch_size=32,  
        validation_data=(X_test, y_test),
        verbose=0
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("\nNajlepsze parametry:")
    print(f"Units: {best_hps.get('units')}")
    print(f"Learning rate: {best_hps.get('learning_rate')}")

    return best_model

def main(run_baseline=True, run_dnn=False, use_tuner=False):
    X, y = load_data()
    X_train, X_test, y_train, y_test, scaler = prepare_data(X, y)
    if run_baseline:
        baseline_model = train_baseline_model(X_train, y_train)
        evaluate_model(baseline_model, X_test, y_test, model_name="RandomForest")
    if run_dnn:
        dnn_model = train_dnn_model(X_train, y_train, X_test, y_test)
        evaluate_model(dnn_model, X_test, y_test, model_name="DNN")
    if use_tuner:
        best_model = run_tuner(X_train, y_train, X_test, y_test, max_trials=10, epochs=50)
        evaluate_model(best_model, X_test, y_test, model_name="DNN (Tuned)")

        
if __name__ == "__main__":
    main(
        run_baseline=False, 
        run_dnn=False, 
        use_tuner=True
        )