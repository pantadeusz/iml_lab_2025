import os
from sys import argv

import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
import tensorflow as tf


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history["val_" + metric], "")
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, "val_" + metric])


def get_data(buffer_size=10000, batch_size=64):

    dataset, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset["train"], dataset["test"]

    train_dataset = (
        train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    )
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


def show_example(encoder, example):

    vocab = np.array(encoder.get_vocabulary())

    encoded_example = encoder(example)[:3].numpy()

    for n in range(3):
        print("Original: ", example[n].numpy())
        print("Round-trip: ", " ".join(vocab[encoded_example[n]]))
        print()


def create_encoder(train_dataset, vocab_size=1000):
    encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    return encoder


def create_model(encoder):
    model = tf.keras.Sequential(
        [
            encoder,
            tf.keras.layers.Embedding(
                input_dim=len(encoder.get_vocabulary()), output_dim=64, mask_zero=True
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    )

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(1e-4),
        metrics=["accuracy"],
    )

    return model


def show_loss_accuracy(model, test_dataset):
    test_loss, test_acc = model.evaluate(test_dataset)
    print("Test Loss:", test_loss)
    print("Test Accuracy:", test_acc)


def save_loss_accuracy_plot(history, save_name="loss_accuracy.png"):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, "accuracy")
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, "loss")
    plt.ylim(0, None)
    plt.savefig(save_name)
    plt.close()

def save_model(model, name):
    if not (os.path.exists("./models")):
        os.mkdir("./models")

    extension = ".keras"
    counter = 0
    base_path = os.path.join("models")
    save_model_path = os.path.join(base_path, f"{name}_{counter}{extension}")

    while os.path.exists(save_model_path):
        save_model_path = os.path.join(base_path, f"{name}_{counter}{extension}")
        counter += 1

    model.save(save_model_path)


def main():
    train_dataset, test_dataset = get_data()
    example = None

    for ex, _ in train_dataset.take(1):
        example = ex
        break

    encoder = create_encoder(train_dataset)
    show_example(encoder, example)

    model = create_model(encoder)
    print([layer.supports_masking for layer in model.layers])

    history = model.fit(
        train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30
    )

    show_loss_accuracy(model, test_dataset)
    save_loss_accuracy_plot(history)

    sample_positive_text = (
        "The movie was cool. The animation and the graphics "
        "were out of this world. I would recommend this movie."
    )
    sample_negative_text = (
        "The movie awful. Waste of time and money."
        "The director should consider different career path."
    )

    predictions = model.predict(
        tf.constant([sample_positive_text, sample_negative_text], dtype=tf.string)
    )
    print(predictions)

    history = model.fit(
        train_dataset, epochs=10, validation_data=test_dataset, validation_steps=30
    )

    show_loss_accuracy(model, test_dataset)
    save_loss_accuracy_plot(history, "loss_accuracy_2.png")

    sample_text = (
        "The movie was not good. The animation and the graphics "
        "were terrible. I would not recommend this movie."
    )
    predictions = model.predict(tf.constant([sample_text], dtype=tf.string))
    print(predictions)

    if len(argv) > 1:
        save_model(model, argv[1])


if __name__ == "__main__":
    tfds.disable_progress_bar()
    main()
