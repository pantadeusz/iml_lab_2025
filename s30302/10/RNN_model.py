import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
from keras_tuner import RandomSearch
tfds.disable_progress_bar()

def load_dataset(BATCH_SIZE, BUFFER_SIZE):
    dataset, info = tfds.load(
        'imdb_reviews',
        with_info=True,
        as_supervised=True
    )

    train_dataset, test_dataset = dataset['train'], dataset['test']

    train_dataset = (train_dataset
                     .shuffle(BUFFER_SIZE)
                     .batch(BATCH_SIZE)
                     .prefetch(tf.data.AUTOTUNE))

    test_dataset = (test_dataset
                    .batch(BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE))

    return train_dataset, test_dataset


def build_encoder(VOCAB_SIZE, train_dataset):
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    return encoder


def build_RNN(hp, encoder):
    model = tf.keras.Sequential()
    model.add(encoder)
    model.add(
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=hp.Choice('embedding_dim', [32, 64, 128]),
            mask_zero=True)
    )

    lstm_layers = hp.Int('lstm_layers', 1, 3)

    for i in range(lstm_layers):
        units = hp.Choice(f'lstm_units_{i}', [32, 64, 128])

        return_seq = (i < lstm_layers - 1)

        model.add(
            tf.keras.layers.Bidirectional(
                tf.keras.layers.LSTM(
                    units,
                    return_sequences=return_seq,
                    dropout=0.3,
                    recurrent_dropout=0.3
                )
            )
        )

    model.add(tf.keras.layers.Dense(
        units=hp.Choice('dense_units', [32, 64, 128, 256]),
        activation=hp.Choice('dense_activation', ['relu', 'tanh'])
    ))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')
    )

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model


def train_RNN(train_dataset, test_dataset, encoder):
    tuner = RandomSearch(
        lambda hp: build_RNN(hp, encoder),
        objective='val_accuracy',
        max_trials=15,
        executions_per_trial=2,
        directory='tuner_dir',
        project_name='rnn_tuning',
    )

    tuner.search(
        train_dataset,
        validation_data=test_dataset,
        epochs=15,
        verbose=1
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = build_RNN(best_hps, encoder)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True
    )

    history = best_model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=30,
        callbacks=[early_stop],
        verbose=1
    )

    test_loss, test_acc = best_model.evaluate(test_dataset)
    best_model.save('model_RNN.keras')

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    return history, test_loss, test_acc


def plot_history(history):
    def plot_metric(metric):
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend([metric, 'val_' + metric])

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_metric('accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_metric('loss')
    plt.ylim(0, None)
    plt.show()


if __name__ == '__main__':
    train_dataset, test_dataset = load_dataset(64, 10000)
    encoder = build_encoder(1000, train_dataset)
    history, test_loss, test_acc = train_RNN(train_dataset, test_dataset, encoder)
    plot_history(history)