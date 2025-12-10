import tensorflow_datasets as tfds
import tensorflow as tf

import matplotlib.pyplot as plt



def load_data():
    dataset, info = tfds.load('imdb_reviews', with_info=True,
                              as_supervised=True)

    train_dataset, test_dataset = dataset['train'], dataset['test']
    return  train_dataset, test_dataset


def prepare_data(train_dataset, test_dataset):
    BUFFER_SIZE = 10000 # Ustawiamy współczynnik do mieszania (Shuffle). Im większy tym mieszanie jest bardziej losowe
    BATCH_SIZE = 64

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset


def create_vocab(train_ds):
    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_ds.map(lambda text, label: text))

    return encoder


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



def train_model(model, train_ds, test_ds):
    history = model.fit(train_ds, epochs=10,
                        validation_data=test_ds,
                        validation_steps=30)

    test_loss, test_acc = model.evaluate(test_ds)

    model.save("sentiment_model.keras")

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)

    return history


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])


def plot_losses(history):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)

def predict_text_sentiment(model ,text):
    return model.predict(tf.constant([text], dtype=tf.string))


if __name__ == '__main__':
    train_ds, test_ds = load_data()
    train_ds, test_ds = prepare_data(train_ds, test_ds)

    encoder = create_vocab(train_ds)
    model = build_model(encoder)

    history = train_model(model, train_ds, test_ds)
    plot_losses(history)