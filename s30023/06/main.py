import tensorflow as tf
import keras_tuner as kt
import tensorflow_datasets as tfds
import numpy as np
from sklearn.metrics import classification_report

IMG_SIZE = 100

def image_normalization(image, label):
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    image = image/255.0
    return image, label

def load_data():
    (ds_train, ds_val, ds_test), ds_info = tfds.load(
        'beans',
        split=['train', 'validation', 'test'],
        shuffle_files=True,
        as_supervised=True,  # (image, label)
        with_info=True,
    )

    batch_size = 32

    ds_train = ds_train.map(image_normalization, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.cache().shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    ds_val = ds_val.map(image_normalization, num_parallel_calls=tf.data.AUTOTUNE)
    ds_val = ds_val.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    ds_test = ds_test.map(image_normalization, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

    return ds_train, ds_val, ds_test, ds_info

def build_model(hp, input_shape=(IMG_SIZE, IMG_SIZE, 3), output_units=3):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))
    model.add(tf.keras.layers.Flatten())

    hp_activation = hp.Choice(
        'activation',
        values=['relu', 'tanh', 'elu']
    )
    hp_initializer = hp.Choice(
        'initializer',
        values=['glorot_uniform', 'he_normal', 'random_normal']
    )

    model.add(
        tf.keras.layers.Dense(
            units=hp.Int("units_1", min_value=32, max_value=256, step=32),
            activation=hp_activation,
            kernel_initializer=hp_initializer
        )
    )

    model.add(
        tf.keras.layers.Dense(
            units=hp.Int("units_2", min_value=16, max_value=128, step=16),
            activation=hp_activation,
            kernel_initializer=hp_initializer
        )
    )

    model.add(tf.keras.layers.Dense(output_units, activation='softmax'))

    hp_learning_rate = hp.Choice("lr", values=[0.0001, 0.001, 0.01])

    hp_optimizer = hp.Choice(
        'optimizer',
        values=['adam', 'sgd', 'rmsprop']
    )

    if hp_optimizer == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=hp_learning_rate)
    elif hp_optimizer == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=hp_learning_rate)
    else:
        optimizer = tf.keras.optimizers.RMSprop(learning_rate=hp_learning_rate)

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    return model


def evaluate_model(model, test_ds, target_names):
    y_pred_prob = model.predict(test_ds)
    y_pred = y_pred_prob.argmax(axis=1)

    # prawdziwe etykiety z obiektu Dataset
    y_test = np.concatenate([y for x, y in test_ds], axis=0)
    print('y_test: \n', y_test)

    print(
        f"""
        --- EVALUATION ---
        {classification_report(y_test, y_pred, target_names=target_names)}
        """)


def save_model(model, path='tensorflow_model_beans.keras'):
    model.save(path)
    print(f'Model saved to {path}')


def main():
    train_ds, val_ds, test_ds, info = load_data()

    num_classes = info.features['label'].num_classes
    target_names = info.features['label'].names
    input_shape = (IMG_SIZE, IMG_SIZE, 3)

    hypermodel_fn = lambda hp: build_model(
        hp,
        input_shape=input_shape,
        output_units=num_classes
    )

    tuner = kt.RandomSearch(
        hypermodel=hypermodel_fn,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=1,
        overwrite=True,
        directory="Beans_model",
        project_name="Lab6",
    )

    tuner.search_space_summary()

    tuner.search(train_ds, epochs=5, validation_data=val_ds)

    best_hyper_param = tuner.get_best_hyperparameters(1)[0]

    model = build_model(
        best_hyper_param,
        input_shape=input_shape,
        output_units=num_classes
    )

    model.fit(train_ds, epochs=10, validation_data=val_ds)

    print(
        f"""
        --- Best hyperparameters: --- 
        - Activation: {best_hyper_param.get('activation')}
        - Initialization method: {best_hyper_param.get('initializer')}
        - Units in hidden layer 1: {best_hyper_param.get('units_1')}
        - Units in hidden layer 2: {best_hyper_param.get('units_2')}
        - Optimizer: {best_hyper_param.get('optimizer')}
        - Learning rate: {best_hyper_param.get('lr')}
        """
    )

    evaluate_model(model, test_ds, target_names=target_names)

    save_model(model)

if __name__ == '__main__':
    main()