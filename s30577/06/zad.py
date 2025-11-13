import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras import layers
import keras_tuner as kt

def load_beans_data():
    (train_ds, val_ds, test_ds), info = tfds.load('beans',split=['train', 'validation', 'test'],with_info=True,as_supervised=True )
    
    num_classes = info.features['label'].num_classes
    print(f"Liczba klas: {num_classes}")
    print(f"Nazwy klas: {info.features['label'].names}")
    
    return train_ds, val_ds, test_ds, num_classes

def preprocess_image(image, label):
    image = tf.image.resize(image, [128, 128])
    image = tf.cast(image, tf.float32) / 255.0 
    return image, label

def prepare_dataset(ds, batch_size=32, shuffle=True):
    ds = ds.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)  
    return ds

def create_model(input_shape, num_classes, initializer='glorot_uniform', activation='relu', optimizer='adam'):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Flatten(),  
        
        layers.Dense(256, activation=activation, kernel_initializer=initializer),
        layers.Dropout(0.3),  
        
        layers.Dense(128, activation=activation, kernel_initializer=initializer),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')  
    ])
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    
    return model

def train_model(model, train_ds, val_ds, epochs=30):
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,  
        restore_best_weights=True
    )
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[early_stop],
        verbose=1
    )
    
    return model, history

def build_tuner_model(input_shape, num_classes):
    def build_model(hp):
        initializer = hp.Choice('initializer', ['glorot_uniform', 'he_normal', 'random_normal'])
        
        activation = hp.Choice('activation', ['relu', 'tanh', 'elu'])
        
        optimizer_name = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
        
        units_1 = hp.Int('units_1', min_value=128, max_value=512, step=128)
        units_2 = hp.Int('units_2', min_value=64, max_value=256, step=64)
        
        model = keras.Sequential([
            layers.Input(shape=input_shape),
            layers.Flatten(),
            layers.Dense(units_1, activation=activation, kernel_initializer=initializer),
            layers.Dropout(0.3),
            layers.Dense(units_2, activation=activation, kernel_initializer=initializer),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=optimizer_name,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    return build_model

def run_tuner(train_ds, val_ds, input_shape, num_classes, max_trials=10, epochs=20):
    build_model = build_tuner_model(input_shape, num_classes)
    
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=max_trials, 
        directory='tuner_results_beans',
        project_name='beans_classifier',
        overwrite=True
    )
    
    print(f"Rozpoczynam przeszukiwanie {max_trials} kombinacji hiperparametrów...")
    
    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    
    print("\n=== NAJLEPSZE PARAMETRY ===")
    print(f"Initializer: {best_hps.get('initializer')}")
    print(f"Activation: {best_hps.get('activation')}")
    print(f"Optimizer: {best_hps.get('optimizer')}")
    print(f"Units warstwa 1: {best_hps.get('units_1')}")
    print(f"Units warstwa 2: {best_hps.get('units_2')}")
    
    best_model = tuner.get_best_models(num_models=1)[0]
    
    return best_model, best_hps

def main():
    print("=== Ładowanie danych Beans ===")
    train_ds, val_ds, test_ds, num_classes = load_beans_data()
    
    print("\n=== Preprocessing danych ===")
    train_prepared = prepare_dataset(train_ds, batch_size=32, shuffle=True)
    val_prepared = prepare_dataset(val_ds, batch_size=32, shuffle=False)
    test_prepared = prepare_dataset(test_ds, batch_size=32, shuffle=False)
    
    input_shape = (128, 128, 3) 
    
    print("\n=== Optymalizacja hiperparametrów ===")
    best_model, best_hps = run_tuner(
        train_prepared, 
        val_prepared, 
        input_shape, 
        num_classes,
        max_trials=6,  
        epochs=15
    )
    
    print("\n=== Ewaluacja na zbiorze testowym ===")
    test_loss, test_accuracy = best_model.evaluate(test_prepared)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    model_path = 'beans_best_model.keras'
    best_model.save(model_path)
    print(f"\nModel zapisany jako: {model_path}")
    
    return best_model, best_hps, test_accuracy

if __name__ == "__main__":
    main()