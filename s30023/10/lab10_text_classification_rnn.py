import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

tfds.disable_progress_bar()

BUFFER_SIZE = 10000
VOCAB_SIZE = 1000
BATCH_SIZE = 64

def load_data():
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset, info

def create_encoder(train_dataset):
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

    print(f'Size of vocabulary: {len(encoder.get_vocabulary())}')

    return encoder

def create_model(encoder):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True, unroll=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, unroll=True)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(1e-4),
                  metrics=['accuracy'])

    return model

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

def plot_history(history):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.show()

def evaluate_model(model, test_dataset):
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'\nTest Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')

def sample_model_predict(model):
    sample_text = ('The movie was not good. The animation and the graphics '
                   'were terrible. I would not recommend this movie.')
    predictions = model.predict(tf.constant([sample_text]))
    print(f"\nSample tekst: {sample_text}")
    print(f"Result (logit): {predictions[0][0]}")
    print("Positive interpretation" if predictions[0][0] >= 0 else "Negative interpretation")

def save_model_to_disk(model, path='model.keras'):
    model.save(path)
    print(f"Model saved to {path} successfully.")

def main():
    train_dataset, test_dataset, info = load_data()
    encoder = create_encoder(train_dataset)
    model = create_model(encoder)

    history = model.fit(train_dataset, epochs=10,
                        validation_data=test_dataset)

    plot_history(history)
    evaluate_model(model, test_dataset)
    sample_model_predict(model)
    save_model_to_disk(model)

if __name__ == '__main__':
    main()