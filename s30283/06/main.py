import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
from typing import Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

DatasetType = tf.data.Dataset

def load_data(dataset_name: str) -> Tuple[DatasetType, DatasetType, DatasetType]:
    (train_ds, val_ds, test_ds) = tfds.load(
        dataset_name,
        split=['train', 'validation', 'test'],
        shuffle_files=True,
        as_supervised=True
    )
    return train_ds, val_ds, test_ds

def all_images_same_size(
    train_ds: DatasetType,
    val_ds: DatasetType,
    test_ds: DatasetType
) -> bool:
    return all(
        [
            all(item[0].shape == (500, 500, 3) for item in ds)
            for ds in [train_ds, val_ds, test_ds]
        ]
    )

def preprocess_image(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32)
    return image / 255.0, label

def get_pipelines(
    train_ds: DatasetType,
    val_ds: DatasetType,
    test_ds: DatasetType,
    batch_size=32
) -> Tuple[DatasetType, DatasetType, DatasetType]:
    '''
    - every ds holds a list of file paths on disk like /.../img.jpg, so it does not use RAM
    - AUTOTUNE enables working in parallel
    - shuffling uses a buffer of 1024 items
    - .batch() packs items together into batches
    - .prefetch() allows CPU/GPU overlap
    '''

    AUTOTUNE = tf.data.AUTOTUNE

    train_ds_pipeline = train_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    train_ds_pipeline = train_ds_pipeline.shuffle(1024)
    train_ds_pipeline = train_ds_pipeline.batch(batch_size)
    train_ds_pipeline = train_ds_pipeline.prefetch(AUTOTUNE)

    val_ds_pipeline = val_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    val_ds_pipeline = val_ds_pipeline.batch(batch_size)
    val_ds_pipeline = val_ds_pipeline.prefetch(AUTOTUNE)

    test_ds_pipeline = test_ds.map(preprocess_image, num_parallel_calls=AUTOTUNE)
    test_ds_pipeline = test_ds_pipeline.batch(batch_size)
    test_ds_pipeline = test_ds_pipeline.prefetch(AUTOTUNE)

    return train_ds_pipeline, val_ds_pipeline, test_ds_pipeline

def build_model(
    hp: kt.HyperParameters,
    input_shape: Tuple[int, int, int]
) -> tf.keras.Sequential:

    activation_tuner = hp.Choice('activation',
                              values=['relu', 'tanh', 'sigmoid', 'elu', 'selu'])
    initializer_tuner = hp.Choice('initializer',
                               values=['glorot_uniform', 'he_normal', 'he_uniform', 'lecun_normal'])
    optimizer_tuner = hp.Choice('optimizer',
                                values=['adam', 'rmsprop', 'sgd', 'adamw'])

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Flatten(input_shape=input_shape))

    model.add(
        tf.keras.layers.Dense(
            units=hp.Int('units_layer1', min_value=64, max_value=128, step=8),
            activation=activation_tuner,
            kernel_initializer=initializer_tuner
        )
    )

    model.add(
        tf.keras.layers.Dense(
            units=hp.Int('units_layer2', min_value=32, max_value=128, step=4),
            activation=activation_tuner,
            kernel_initializer=initializer_tuner
        )
    )

    model.add(
        tf.keras.layers.Dense(
            units=hp.Int('units_layer3', min_value=4, max_value=32, step=4),
            activation=activation_tuner,
            kernel_initializer=initializer_tuner
        )
    )

    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    learning_rate = hp.Float('lr', min_value=1e-4, max_value=1e-2, sampling='log')

    if optimizer_tuner == 'adam':
        final_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_tuner == 'adamw':
        final_optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
    elif optimizer_tuner == 'sgd':
        final_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_tuner == 'rmsprop':
        final_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)

    model.compile(
        optimizer=final_optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_and_train_nn_model(
    train_ds: DatasetType,
    val_ds: DatasetType,
    input_shape: Tuple[int, int, int],
    num_models_to_test: int,
    epochs_per_model: int,
    patience: int
) -> tf.keras.Sequential:

    model_builder = lambda hp: build_model(hp,
                                           input_shape=input_shape )

    tuner = kt.RandomSearch(
        model_builder,
        objective='val_accuracy',
        max_trials=num_models_to_test,
        executions_per_trial=1,
        directory='tuning_dir',
        project_name='beans_tuning'
    )

    stop_early = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )

    print('Starting hyperparameter search...')

    tuner.search(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_per_model,
        callbacks=[stop_early],
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f'''
        Best hyperparameters:
        - Layer 1 Units: {best_hps.get('units_layer1')}
        - Layer 2 Units: {best_hps.get('units_layer2')}
        - Layer 3 Units: {best_hps.get('units_layer3')}
        - Activation: {best_hps.get('activation')}
        - Optimizer: {best_hps.get('optimizer')}
        - Initializer: {best_hps.get('initializer')}
        - Learning Rate: {best_hps.get('lr'):.5f}
        '''
    )

    best_model = tuner.get_best_models(num_models=1)[0]
    return best_model, tuner

def generate_plots(tuner: kt):
    results = []
    for trial in tuner.oracle.get_best_trials(num_trials=9999):
        results.append({**trial.hyperparameters.values,
                        'val_acc': trial.score})

    df = pd.DataFrame(results)

    sns.set_palette('rocket')
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))
    sns.barplot(data=df, x='activation', y='val_acc', ax=axes[0])
    axes[0].set_title('Activations Comparison')
    axes[0].set_xlabel('Activation Function')
    axes[0].set_ylabel('Validation Accuracy')

    sns.barplot(data=df, x='optimizer', y='val_acc', ax=axes[1])
    axes[0].set_title('Optimizers Comparison')
    axes[0].set_xlabel('Optimizer')
    axes[0].set_ylabel('Validation Accuracy')

    sns.barplot(data=df, x='initializer', y='val_acc', ax=axes[2])
    axes[0].set_title('Initializers Comparison')
    axes[0].set_xlabel('Initializer')
    axes[0].set_ylabel('Validation Accuracy')

    plt.savefig('basic.png')

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.barplot(
        data=df,
        x='activation',
        y='val_acc',
        hue='optimizer',
        ax=axes[0]
    )
    axes[0].set_title('Activation vs Optimizer')
    axes[0].set_xlabel('Activation Function')
    axes[0].set_ylabel('Validation Accuracy')
    sns.barplot(
        data=df,
        x='initializer',
        y='val_acc',
        hue='optimizer',
        ax=axes[1]
    )
    axes[1].set_title('Initializer vs Optimizer')
    axes[1].set_xlabel('Initializer')
    axes[1].set_ylabel('Validation Accuracy')

    plt.savefig('advanced.png')

if __name__ == '__main__':
    dataset_name = 'beans'
    batch_size = 32

    train_ds, val_ds, test_ds = load_data(dataset_name)

    if not all_images_same_size(train_ds, val_ds, test_ds):
        raise Exception('Images do not have the same size.')

    train_ds_pipeline, val_ds_pipeline, test_ds_pipeline = get_pipelines(
        train_ds, val_ds, test_ds, batch_size
    )

    best_model, tuner = tune_and_train_nn_model(
        train_ds_pipeline,
        val_ds_pipeline,
        input_shape=(128, 128, 3),
        num_models_to_test=100,
        epochs_per_model=100,
        patience=10
    )

    generate_plots(tuner)

    save_path = './best_model.keras'
    print('Best model summary:')
    best_model.summary()
    best_model.save(save_path)
    print('Model has been saved: {}'.format(save_path))