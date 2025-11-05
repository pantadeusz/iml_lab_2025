# ========================
# 1. Import & Setup
# ========================
from sklearn.utils.validation import validate_data
import tensorflow as tf
import keras
import keras_tuner as kt
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time
import numpy as np

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)


# ========================
# 1. Load and Preprocess Data
# ========================
def load_and_preprocess_mnist():
    """
    Load MNIST and normalize pixel values.
    Returns: (x_train, y_train), (x_test, y_test)
    """
    print("📥 Loading and preprocessing MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    return (x_train, y_train), (x_test, y_test)


# ========================
# 2. Split Data (Train/Val/Test)
# ========================
def split_data(x_train, y_train, test_size=0.1, random_state=42):
    """
    Split training data into train and validation.
    Returns: x_train, x_val, y_train, y_val
    """
    return train_test_split(x_train, y_train, test_size=test_size, random_state=random_state)


# ========================
# 3. Universal Plot Function (Works for Any Model!)
# ========================
def plot_training_history(history, title="Training History", model_name="Model", color='blue'):
    """
    Plot training and validation accuracy & loss.
    Works for:
        - dict: Keras history (with 'accuracy', 'val_accuracy', etc.)
        - float: Sklearn accuracy (no epoch history)
        - None: Tuner model (we'll pass the best model's history)
    """
    if hasattr(history, "history"):
        history = history.history

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Accuracy
    if isinstance(history, dict):
        # Keras: plot epoch-by-epoch
        ax1.plot(history['accuracy'], label=f'{model_name} Training', color=color, linewidth=2.5)
        ax1.plot(history['val_accuracy'], label=f'{model_name} Validation', color=color, linestyle='--', linewidth=2.5)
        test_acc = history['val_accuracy'][-1]  # Final val accuracy
    else:
        # Sklearn or tuner: just show final accuracy
        ax1.axhline(y=history, color=color, linestyle='-', linewidth=2.5, label=f'{model_name} Accuracy: {history:.4f}')
        test_acc = history

    ax1.set_title('Accuracy', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(fontsize=12)
    ax1.grid(True, alpha=0.3)

    # Loss
    if isinstance(history, dict):
        ax2.plot(history['loss'], label=f'{model_name} Training Loss', color=color, linewidth=2.5)
        ax2.plot(history['val_loss'], label=f'{model_name} Validation Loss', color=color, linestyle='--', linewidth=2.5)
        test_loss = history['val_loss'][-1]
    else:
        # Approximate loss from accuracy
        approx_loss = -np.log(test_acc) if test_acc > 0 else 10.0
        ax2.axhline(y=approx_loss, color=color, linestyle='-', linewidth=2.5, label=f'{model_name} Loss: {approx_loss:.3f}')
        test_loss = approx_loss

    ax2.set_title('Loss', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Final layout
    plt.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(f"{model_name}_training_plot.png")
    plt.close()


# ========================
# 4. Universal Train & Evaluate Function
# ========================
def train_and_evaluate(model, x_train, y_train, x_val, y_val, model_type="keras", epochs=10, batch_size=32):
    """
    Train and evaluate any model type.

    Args:
        model: Model object (Keras, sklearn, or tuner)
        x_train, y_train: Training data
        x_val, y_val: Validation data
        model_type: "keras", "sklearn", or "tuner"
        epochs: Training epochs
        batch_size: Batch size
        tuner: Optional tuner object (for tuner models)

    Returns:
        history: dict (Keras), accuracy (sklearn), or None
        test_acc: Final test accuracy
    """
    start_time = time.time()

    if model_type == "keras":
        print(f"🧠 Training Keras model for {epochs} epochs...")
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=0
        )
        test_loss, test_acc = model.evaluate(x_val, y_val, verbose=0)
        training_time = time.time() - start_time
        print(f"✅ Keras Training Time: {training_time:.2f}s")
        return history, test_acc

    elif model_type == "sklearn":
        print("🧠 Training Scikit-learn MLP...")
        x_train_flat = x_train.reshape(x_train.shape[0], -1)
        x_val_flat = x_val.reshape(x_val.shape[0], -1)
        model.fit(x_train_flat, y_train)
        y_pred = model.predict(x_val_flat)
        test_acc = accuracy_score(y_val, y_pred)
        training_time = time.time() - start_time
        print(f"✅ Sklearn Training Time: {training_time:.2f}s")
        return test_acc, test_acc  # history = accuracy

    else:
        raise ValueError("model_type must be 'keras' or 'sklearn'")


# ========================
# 5. Build Models
# ========================
def build_keras_model(hp, input_shape=(28, 28)):
    """
    Keras model for tuner.
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(
            units=hp.Int('units', min_value=32, max_value=256, step=32),
            activation='relu'
        ),
        keras.layers.Dropout(
            rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
        ),
        keras.layers.Dense(10)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def build_keras_baseline(hidden_units=128, dropout_rate=0.2):
    """
    Baseline Keras model.
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(hidden_units, activation='relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(10)
    ])
    model.compile(
        optimizer='adam',
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy']
    )
    return model


def build_sklearn_mlp(hidden_units=128, alpha=1e-4, max_iter=200):
    """
    Scikit-learn MLP model.
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes=(hidden_units,),
            activation='relu',
            solver='adam',
            alpha=alpha,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            n_iter_no_change=10
        ))
    ])


# ========================
# 6. Main: Run All Models & Compare
# ========================
import joblib
import os

def run_comparison(
    epochs=10,
    batch_size=32,
    test_size=0.1,
    random_state=42,
    cache_dir="model_cache"
):
    """
    Run and compare models with safe caching.
    Now handles: model, hyperparameters, and training history.
    """
    print("🚀 Starting Full Model Comparison: Keras vs Sklearn vs Keras Tuner")
    print("="*70)

    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)

    # File paths
    keras_model_path = os.path.join(cache_dir, "best_keras_model.h5")
    keras_history_path = os.path.join(cache_dir, "keras_history.pkl")  # ← NEW
    sklearn_model_path = os.path.join(cache_dir, "best_sklearn_model.pkl")
    tuner_hp_path = os.path.join(cache_dir, "best_hyperparameters.pkl")
    tuner_model_path = os.path.join(cache_dir, "best_tuner_model.h5")
    tuner_results_path = os.path.join(cache_dir, "tuner_results.pkl")

    # Step 1: Load data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_mnist()
    x_train, x_val, y_train, y_val = split_data(x_train, y_train, test_size=test_size, random_state=random_state)

    # Step 2: Try to load cached models and history
    print("\n📂 Checking for cached models and training history...")

    # Load Keras model and history
    if os.path.exists(keras_model_path) and os.path.exists(keras_history_path):
        print(f"✅ Loading cached Keras model and history from {keras_model_path} and {keras_history_path}")
        keras_model = keras.models.load_model(keras_model_path)
        keras_history = joblib.load(keras_history_path)
        
        _, keras_acc = keras_model.evaluate(x_val, y_val, verbose=0)
    else:
        print("🆕 Building and training Keras model...")
        keras_model = build_keras_baseline(hidden_units=128, dropout_rate=0.2)
        keras_history, keras_acc = train_and_evaluate(
            keras_model, x_train, y_train, x_val, y_val,
            model_type="keras", epochs=epochs, batch_size=batch_size
        )
        print(f"💾 Saving Keras model to {keras_model_path}")
        keras_model.save(keras_model_path)
        print(f"💾 Saving Keras history to {keras_history_path}")
        joblib.dump(keras_history.history, keras_history_path)

    # Load Sklearn model
    if os.path.exists(sklearn_model_path):
        print(f"✅ Loading cached Sklearn model from {sklearn_model_path}")
        sklearn_model = joblib.load(sklearn_model_path)
        # 👇 Add this to compute accuracy
        x_val_flat = x_val.reshape(x_val.shape[0], -1)
        y_pred = sklearn_model.predict(x_val_flat)
        sklearn_acc = accuracy_score(y_val, y_pred)
    else:
        print("🆕 Building and training Sklearn MLP...")
        sklearn_model = build_sklearn_mlp(hidden_units=128, alpha=1e-4)
        sklearn_acc = train_and_evaluate(
            sklearn_model, x_train, y_train, x_val, y_val,
            model_type="sklearn", epochs=epochs, batch_size=batch_size
        )[0]
        print(f"💾 Saving Sklearn model to {sklearn_model_path}")
        joblib.dump(sklearn_model, sklearn_model_path)

    # Load Tuner Results
    if os.path.exists(tuner_model_path) and os.path.exists(tuner_hp_path):
        print(f"✅ Loading cached Tuner model and hyperparameters...")
        best_hp = joblib.load(tuner_hp_path)
        best_model = keras.models.load_model(tuner_model_path)
        tuner_model = build_keras_model(best_hp)
        tuner_model.set_weights(best_model.get_weights())
        tuner_history, tuner_acc = train_and_evaluate(
            tuner_model, x_train, y_train, x_val, y_val,
            model_type="keras", epochs=epochs, batch_size=batch_size
        )
    else:
        print("🆕 Running Keras Tuner hyperparameter search...")
        tuner = kt.RandomSearch(
            build_keras_model,
            objective='val_accuracy',
            max_trials=10,
            directory='tuner_results',
            project_name='mnist_tuner'
        )
        tuner.search(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_val, y_val),
            verbose=1
        )

        best_hp = tuner.get_best_hyperparameters()[0]
        best_model = tuner.get_best_models(num_models=1)[0]
        tuner_model = build_keras_model(best_hp)
        tuner_model.set_weights(best_model.get_weights())

        tuner_history, tuner_acc = train_and_evaluate(
            tuner_model, x_train, y_train, x_val, y_val,
            model_type="keras", epochs=epochs, batch_size=batch_size
        )

        # Save only best hyperparameters and model
        print(f"💾 Saving best hyperparameters to {tuner_hp_path}")
        joblib.dump(best_hp, tuner_hp_path)
        print(f"💾 Saving best model to {tuner_model_path}")
        best_model.save(tuner_model_path)

        # Save results
        results = {
            'best_hp': best_hp,
            'best_model_path': tuner_model_path,
            'best_accuracy': tuner_acc
        }
        joblib.dump(results, tuner_results_path)

    # Step 3: Plot results
    print("\n📈 Plotting comparison...")
    plot_training_history(keras_history, title="Model Comparison: Keras vs Sklearn vs Tuner", model_name="Keras", color='blue')
    plot_training_history(sklearn_acc, title="Model Comparison: Keras vs Sklearn vs Tuner", model_name="Sklearn", color='green')
    plot_training_history(tuner_history, title="Model Comparison: Keras vs Sklearn vs Tuner", model_name="Keras Tuner", color='purple')

    # Step 4: Summary
    print("\n" + "="*70)
    print("📊 FINAL COMPARISON SUMMARY")
    print("="*70)
    print(f"Keras Model Accuracy:       {keras_acc:.4f}")
    print(f"Scikit-learn MLP Accuracy:  {sklearn_acc:.4f}")
    print(f"Keras Tuner Best Accuracy:  {tuner_acc:.4f}")
    print(f"Best Model: {'Keras Tuner' if tuner_acc > keras_acc and tuner_acc > sklearn_acc else 'Keras' if keras_acc > sklearn_acc else 'Sklearn'}")
    print(f"✅ All models loaded or trained successfully!")

# ========================
# 7. Run It!
# ========================
if __name__ == "__main__":
    run_comparison(epochs=10, batch_size=32)
