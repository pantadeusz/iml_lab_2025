import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt
from pathlib import Path

BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = 1000
EPOCHS = 5
MODEL_DIR = "sentiment_rnn_model_tuned.keras"


def load_raw_datasets():
    datasets, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    return datasets["train"], datasets["test"]


def create_encoder(raw_train_ds):
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(raw_train_ds.map(lambda text, label: text))
    return encoder


def prepare_tf_datasets(raw_train_ds, raw_test_ds):
    train_ds = (
        raw_train_ds.shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        raw_test_ds.batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds


def build_model(hp, encoder):
    embedding_dim = hp.Choice("embedding_dim", [16, 32, 64])
    lstm_units_1 = hp.Choice("lstm_units_1", [32, 64, 96])
    lstm_units_2 = hp.Choice("lstm_units_2", [16, 32, 64])
    dropout_rate = hp.Choice("dropout", [0.2, 0.3, 0.4, 0.5])
    learning_rate = hp.Choice("learning_rate", [1e-3, 1e-4])

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=embedding_dim,
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units_1, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(lstm_units_2)
        ),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1)
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"]
    )

    return model


def main():
    raw_train_ds, raw_test_ds = load_raw_datasets()
    encoder = create_encoder(raw_train_ds)
    train_ds, test_ds = prepare_tf_datasets(raw_train_ds, raw_test_ds)

    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, encoder),
        objective="val_accuracy",
        max_trials=10,
        overwrite=True,
        directory="keras_tuner_results",
        project_name="sentiment_rnn"
    )

    tuner.search(
        train_ds,
        epochs=EPOCHS,
        validation_data=test_ds,
        validation_steps=20
    )

    print("najlepszy model znaleziony przez tuner:")
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save(MODEL_DIR)


if __name__ == "__main__":
    main()
