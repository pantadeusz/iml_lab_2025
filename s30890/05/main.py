import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt
import time


def load_data():
    """Load and split the breast cancer dataset"""
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, data.feature_names


def preprocess_data(X_train, X_test):
    """Standardize features"""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler


# baseline model using sklearn
def train_baseline_model(X_train, y_train):
    """Train baseline logistic regression model"""
    model = LogisticRegression(random_state=42, max_iter=3000)
    model.fit(X_train, y_train)
    return model


def evaluate_baseline_model(model, X_test, y_test):
    """Evaluate baseline model and return predictions"""
    y_pred = model.predict(X_test)
    return y_pred


# baseline DNN model (keras)
def create_baseline_dnn(input_dim):
    """Create baseline DNN model with fixed architecture"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model


def train_dnn_model(model, X_train, y_train, X_val, y_val, epochs=50, verbose=0):
    """Train DNN model"""
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        verbose=verbose
    )
    return history


def evaluate_dnn_model(model, X_test, y_test):
    """Evaluate DNN model and return predictions"""
    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = (y_pred_proba > 0.5).astype(int).flatten()
    return y_pred


# keras tuner
def build_tunable_model(hp):
    """Build model with hyperparameters to tune"""
    model = keras.Sequential()
    
    # Tunable first layer
    hp_units_1 = hp.Int('units_layer_1', min_value=32, max_value=256, step=32)
    model.add(layers.Dense(hp_units_1, activation='relu', input_shape=(30,)))
    
    # Tunable dropout
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.0, max_value=0.5, step=0.1)
    model.add(layers.Dropout(hp_dropout_1))
    
    # Tunable second layer
    hp_units_2 = hp.Int('units_layer_2', min_value=16, max_value=128, step=16)
    model.add(layers.Dense(hp_units_2, activation='relu'))
    
    # Tunable dropout
    hp_dropout_2 = hp.Float('dropout_2', min_value=0.0, max_value=0.5, step=0.1)
    model.add(layers.Dropout(hp_dropout_2))
    
    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))
    
    # Tunable learning rate
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    
    return model


def tune_model(X_train, y_train, X_val, y_val, max_trials=10, executions_per_trial=1):
    """Run hyperparameter tuning"""
    tuner = kt.RandomSearch(
        build_tunable_model,
        objective='val_accuracy',
        max_trials=max_trials,
        executions_per_trial=executions_per_trial,
        directory='tuner_results',
        project_name='breast_cancer_tuning',
        overwrite=True
    )
    
    print(f"\n{'='*60}")
    print("Starting hyperparameter tuning...")
    print(f"Max trials: {max_trials}")
    print(f"{'='*60}\n")
    
    tuner.search(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        verbose=0
    )
    
    return tuner


# results display
def display_results(name, y_test, y_pred):
    """Display classification report and confusion matrix"""
    print(f"\n{'='*60}")
    print(f"{name.upper()} RESULTS")
    print(f"{'='*60}\n")
    
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(pd.DataFrame(cm, 
                       index=['Actual 0 (malignant)', 'Actual 1 (benign)'],
                       columns=['Pred 0', 'Pred 1']))
    
    print("\nClassification Report:")
    report = classification_report(y_test, y_pred, target_names=['malignant', 'benign'])
    print(report)
    
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    return report_dict


# execution
def main():
    print("Dataset: Breast Cancer Wisconsin")
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Task 1: Load and preprocess data
    print("\n[Task 1] Loading data...")
    X_train, X_test, y_train, y_test, feature_names = load_data()
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]}")
    
    X_train_scaled, X_test_scaled, scaler = preprocess_data(X_train, X_test)
    
    # Create validation set
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Task 2: Train and evaluate baseline sklearn model
    print("\n[Task 2] Training baseline scikit-learn model...")
    baseline_sklearn = train_baseline_model(X_train_scaled, y_train)
    y_pred_sklearn = evaluate_baseline_model(baseline_sklearn, X_test_scaled, y_test)
    report_sklearn = display_results("Baseline Scikit-Learn Model", y_test, y_pred_sklearn)
    
    # Train baseline DNN without tuning
    print("\n[Task 2] Training baseline DNN model (without tuning)...")
    baseline_dnn = create_baseline_dnn(input_dim=X_train_scaled.shape[1])
    train_dnn_model(baseline_dnn, X_train_final, y_train_final, X_val, y_val, epochs=50, verbose=0)
    y_pred_dnn_baseline = evaluate_dnn_model(baseline_dnn, X_test_scaled, y_test)
    report_dnn_baseline = display_results("Baseline DNN Model (no tuning)", y_test, y_pred_dnn_baseline)
    
    # Task 3 & 4: Hyperparameter tuning
    print("\n[Task 3 & 4] Running hyperparameter tuning...")
    
    # Estimate time for single trial
    print("\nEstimating time per trial...")
    start_time = time.time()
    test_model = build_tunable_model(kt.HyperParameters())
    test_model.fit(X_train_final, y_train_final, 
                   validation_data=(X_val, y_val),
                   epochs=30, batch_size=32, verbose=0)
    time_per_trial = time.time() - start_time
    print(f"Estimated time per trial: {time_per_trial:.2f} seconds")
    
    # Run tuning (limit to 10 trials for 30 minutes constraint)
    max_trials = 10
    print(f"\nEstimated total time for {max_trials} trials: {(time_per_trial * max_trials / 60):.2f} minutes")
    
    tuner = tune_model(X_train_final, y_train_final, X_val, y_val, 
                      max_trials=max_trials, executions_per_trial=1)
    
    # Get best model
    print("\n[Task 4] Retrieving best model...")
    best_models = tuner.get_best_models(num_models=1)
    best_model = best_models[0]
    
    # Display best hyperparameters
    print("\nBest Hyperparameters:")
    best_hp = tuner.get_best_hyperparameters(1)[0]
    print(f"  Layer 1 units: {best_hp.get('units_layer_1')}")
    print(f"  Dropout 1: {best_hp.get('dropout_1')}")
    print(f"  Layer 2 units: {best_hp.get('units_layer_2')}")
    print(f"  Dropout 2: {best_hp.get('dropout_2')}")
    print(f"  Learning rate: {best_hp.get('learning_rate')}")
    
    # Evaluate tuned model
    y_pred_tuned = evaluate_dnn_model(best_model, X_test_scaled, y_test)
    report_tuned = display_results("Tuned DNN Model (with Keras Tuner)", y_test, y_pred_tuned)
    
    # Task 5: Summary comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    
    comparison_df = pd.DataFrame({
        'Model': ['Scikit-Learn Baseline', 'DNN Baseline', 'DNN Tuned'],
        'Accuracy': [
            report_sklearn['accuracy'],
            report_dnn_baseline['accuracy'],
            report_tuned['accuracy']
        ],
        'Precision (weighted)': [
            report_sklearn['weighted avg']['precision'],
            report_dnn_baseline['weighted avg']['precision'],
            report_tuned['weighted avg']['precision']
        ],
        'Recall (weighted)': [
            report_sklearn['weighted avg']['recall'],
            report_dnn_baseline['weighted avg']['recall'],
            report_tuned['weighted avg']['recall']
        ],
        'F1-Score (weighted)': [
            report_sklearn['weighted avg']['f1-score'],
            report_dnn_baseline['weighted avg']['f1-score'],
            report_tuned['weighted avg']['f1-score']
        ]
    })
    
    print("\n" + comparison_df.to_string(index=False))
    
    return {
        'sklearn': report_sklearn,
        'dnn_baseline': report_dnn_baseline,
        'dnn_tuned': report_tuned,
        'best_hp': best_hp
    }


if __name__ == "__main__":
    results = main()