import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
from pathlib import Path

BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = 1000
EPOCHS = 5
MODEL_DIR = "sentiment_rnn_model"

def load_raw_datasets():
    datasets, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    raw_train_ds, raw_test_ds = datasets['train'], datasets['test']
    return raw_train_ds, raw_test_ds

def create_encoder(raw_train_ds, vocab_size=VOCAB_SIZE):
    encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
    text_ds = raw_train_ds.map(lambda text, label: text)
    encoder.adapt(text_ds)
    return encoder

def prepare_tf_datasets(raw_train_ds, raw_test_ds):
    train_ds = (
        raw_train_ds
        .shuffle(BUFFER_SIZE)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    test_ds = (
        raw_test_ds
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )
    return train_ds, test_ds


def build_rnn_model(encoder):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=32,
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(64, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(32)
        ),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)  # logit
    ])

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"]
    )

    return model

def train_and_evaluate(model, train_ds, test_ds, epochs=EPOCHS):
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=test_ds,
        validation_steps=30
    )

    test_loss, test_acc = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    return history, test_loss, test_acc


def save_model(model, model_dir=MODEL_DIR):
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    model.save("sentiment_rnn_model.keras")
    print(f"model zapisany w katalogu: {model_dir}")


def main():
    #ladowanie surowych danych
    raw_train_ds, raw_test_ds = load_raw_datasets()

    #tworzenie enkodera i adaptowanie go na tekstach treningowych
    encoder = create_encoder(raw_train_ds)

    #przygotowanie pipelinow danych tf.data
    train_ds, test_ds = prepare_tf_datasets(raw_train_ds, raw_test_ds)
    
    model = build_rnn_model(encoder)
    train_and_evaluate(model, train_ds, test_ds, epochs=EPOCHS)
    save_model(model, MODEL_DIR)


if __name__ == "__main__":
    main()