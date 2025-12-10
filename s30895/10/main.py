import numpy as np
import keras_tuner as kt
import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()

import matplotlib.pyplot as plt

BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = 1000


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

def get_data():

    dataset, info = tfds.load('imdb_reviews', with_info=True,                      as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_dataset, test_dataset

train_dataset, test_dataset = get_data()

def get_encoder(train_dataset):
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))
    return encoder

def get_vocab(encoder):
    vocab = np.array(encoder.get_vocabulary())
    return vocab

def build_model(encoder):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    return model

def build_model_tuner(hp):
    encoder = get_encoder(train_dataset)

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            len(encoder.get_vocabulary()),
            hp.Int('embedding_dim', min_value=32, max_value=128, step=32),
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                hp.Int('lstm_1_units', min_value=32, max_value=128, step=32),
                return_sequences=True
            )
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(
                hp.Int('lstm_2_units', min_value=16, max_value=64, step=16)
            )
        ),
        tf.keras.layers.Dense(
            hp.Int('dense_units', min_value=32, max_value=128, step=32),
            activation='relu'
        ),
        tf.keras.layers.Dropout(hp.Float('dropout', 0.2, 0.4, step=0.2)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Float('learning_rate', 1e-4, 1e-3, sampling='log')
        ),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy']
    )

    return model


def fit_and_validate(model):
    history = model.fit(train_dataset, epochs=1,
                        validation_data=test_dataset,
                        validation_steps=30)

    test_loss, test_acc = model.evaluate(test_dataset)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')


def find_best_architecture(build_model_tuner):
    tuner = kt.RandomSearch(
        build_model_tuner,
        objective='val_accuracy',
        max_trials=10,  # Try 5 different hyperparameter sets
        executions_per_trial=1,
        directory='kt_tuner',
        project_name='imdb_sentiment'
    )

    tuner.search_space_summary()

    tuner.search(
        train_dataset,
        epochs=5,
        validation_data=test_dataset
    )
    best_model = tuner.get_best_models(num_models=1)[0]
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print("Best hyperparameters:")
    print(best_hps.values)
    return best_model


encoder = get_encoder(train_dataset)
vocab = get_vocab(encoder)
model = find_best_architecture(build_model_tuner)
fit_and_validate(model)
model.save("model.keras")
