import kagglehub
import os
import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from typing import Tuple

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_dataset() -> str:
    logging.info('🧠 Downloading dataset')
    
    path = kagglehub.dataset_download('fedesoriano/heart-failure-prediction')
    
    csv_path = os.path.join(path, 'heart.csv') 
    
    logging.info(f'✅ Dataset downloaded to: {csv_path}')
    return csv_path

def load_data(path: str) -> pd.DataFrame:
    logging.info('🧠 Loading data')
    
    data = pd.read_csv(path)
    
    logging.info('✅ Data loaded')
    return data

def preprocess_data(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logging.info('🧠 Preprocessing data')

    target = 'HeartDisease'
    X = data.drop([target], axis=1)
    y = data[target]
    
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numerical_cols = X.select_dtypes(include=np.number).columns
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train_encoded = pd.get_dummies(X_train, columns=categorical_cols)
    X_test_encoded = pd.get_dummies(X_test, columns=categorical_cols)
    
    # Handles cases where some categories might only appear in test or train
    X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
    
    scaler = StandardScaler()
    
    X_train_scaled = X_train_encoded.copy()
    X_test_scaled = X_test_encoded.copy()

    num_cols_in_encoded = [col for col in numerical_cols if col in X_train_encoded.columns]

    X_train_scaled[num_cols_in_encoded] = scaler.fit_transform(X_train_encoded[num_cols_in_encoded])
    X_test_scaled[num_cols_in_encoded] = scaler.transform(X_test_encoded[num_cols_in_encoded])
    
    logging.info('✅ Preprocessing finished')
    
    return X_train_scaled.values, X_test_scaled.values, y_train.values, y_test.values

def prepare_loaders(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray):
    logging.info('🧠 Preparing loaders')
    
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))

    BATCH_SIZE = 32
    train_ds = (train_ds
                .shuffle(buffer_size=len(X_train))
                .batch(BATCH_SIZE)
                .prefetch(tf.data.AUTOTUNE))

    test_ds = (test_ds
               .batch(BATCH_SIZE)
               .prefetch(tf.data.AUTOTUNE))
    
    logging.info('✅ Loaders prepared')
    return train_ds, test_ds

def get_random_forest_model(X_train: np.ndarray, y_train: np.ndarray) -> RandomForestClassifier:
    logging.info('🧠 Preparing random forest model')
    
    model = RandomForestClassifier(n_estimators=100, criterion='log_loss', random_state=42)
    model.fit(X_train, y_train)
    
    logging.info('✅ Random forest model prepared')
    return model

def build_model(hp: kt.HyperParameters, input_shape: int) -> Sequential:
    """
    This is the model-building function that KerasTuner will use.
    'hp' is an object that allows you to define tunable hyperparameters.
    """
    model = Sequential()
    model.add(Dense(
        units=hp.Int('units_layer1', min_value=8, max_value=64, step=8),
        activation='relu',
        input_shape=(input_shape,)
    ))
    model.add(Dense(
        units=hp.Int('units_layer2', min_value=4, max_value=32, step=4),
        activation='relu'
    ))
    model.add(Dense(1, activation='sigmoid'))
    
    learning_rate = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_and_train_nn_model(X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Sequential:
    logging.info('🧠 Preparing neural network model and starting tuning')
    
    train_ds, test_ds = prepare_loaders(X_train, X_test, y_train, y_test)
    
    input_shape = X_train.shape[1]

    model_builder = lambda hp: build_model(hp, input_shape=input_shape)

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',  # What to maximize
        max_trials=50,             # How many different models to try
        executions_per_trial=1,    # How many times to train each model
        directory='tuning_dir',
        project_name='heart_failure_tuning'
    )
    
    stop_early = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    logging.info('🚀 Starting hyperparameter search...')
    tuner.search(
        train_ds,
        validation_data=test_ds,
        epochs=150,
        callbacks=[stop_early],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    logging.info(f"""
    ✅ Tuning complete. Best hyperparameters found:
    - Layer 1 Units: {best_hps.get('units_layer1')}
    - Layer 2 Units: {best_hps.get('units_layer2')}
    - Learning Rate: {best_hps.get('lr'):.5f}
    """)

    best_model = tuner.get_best_models(num_models=1)[0]
    
    logging.info('✅ Best neural network model prepared')
    return best_model

if __name__ == '__main__':
    try:
        dataset_path = download_dataset()
        dataframe = load_data(dataset_path)
        
        X_train, X_test, y_train, y_test = preprocess_data(dataframe)
        
        rf_model = get_random_forest_model(X_train, y_train)
        y_pred_rf = rf_model.predict(X_test)
        
        logging.info("\n--- Random Forest Classification Report ---")
        print(classification_report(y_test, y_pred_rf, digits=4))

        nn_model = tune_and_train_nn_model(X_train, X_test, y_train, y_test)
        y_pred_nn_prob = nn_model.predict(X_test.astype(np.float32))
        y_pred_nn = (y_pred_nn_prob > 0.5).astype(int)
        
        logging.info("\n--- Tuned Neural Network Classification Report ---")
        print(classification_report(y_test, y_pred_nn, digits=4))

    except Exception as e:
        logging.error(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()