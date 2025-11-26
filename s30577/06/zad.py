import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt


def preprocess(image, label):
    image = tf.image.resize(image, [128, 128])
    return tf.cast(image, tf.float32) / 255.0, label


(train_ds, val_ds, test_ds), info = tfds.load('beans', split=['train', 'validation', 'test'], with_info=True,
                                              as_supervised=True)

train_ds = train_ds.map(preprocess).cache().shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).batch(32).cache()
test_ds = test_ds.map(preprocess).batch(32)


def create_model(initializer='glorot_uniform', activation='relu', optimizer='adam'):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(156, activation=activation, kernel_initializer=initializer),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation=activation, kernel_initializer=initializer),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation=activation, kernel_initializer=initializer),
        tf.keras.layers.Dense(3, activation='softmax')
    ])

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def model_builder(hp):
    hp_init = hp.Choice('initializer', values=['glorot_uniform', 'he_normal'])
    hp_act = hp.Choice('activation', values=['relu', 'elu', 'tanh'])
    hp_opt = hp.Choice('optimizer', values=['adam', 'sgd', 'rmsprop'])

    return create_model(initializer=hp_init, activation=hp_act, optimizer=hp_opt)


tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=20,
                     factor=3,
                     directory='lab6_tuner_mlp',
                     project_name='beans_mlp_monster')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(train_ds, validation_data=val_ds, callbacks=[stop_early])

best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(
    f"Najlepsze parametry: Act={best_hps.get('activation')}, Opt={best_hps.get('optimizer')}, Init={best_hps.get('initializer')}")

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save('model_beans.keras')