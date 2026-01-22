
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

tfds.disable_progress_bar()



class Config:
    """Parametry konfiguracyjne modelu i treningu."""
    BUFFER_SIZE = 10000
    BATCH_SIZE = 64
    VOCAB_SIZE = 1000
    EMBEDDING_DIM = 64
    LSTM_UNITS = 64
    DENSE_UNITS = 64
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    MODEL_SAVE_PATH = 'sentiment_model.keras'



def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_' + metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_' + metric])


def plot_training_history(history):
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)
    plt.savefig('training_history.png')
    plt.show()



def load_data():
    print("Ładowanie danych IMDB...")
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    return train_dataset, test_dataset, info


def prepare_datasets(train_dataset, test_dataset, config):
    train_dataset = (train_dataset
                     .shuffle(config.BUFFER_SIZE)
                     .batch(config.BATCH_SIZE)
                     .prefetch(tf.data.AUTOTUNE))
    test_dataset = (test_dataset
                    .batch(config.BATCH_SIZE)
                    .prefetch(tf.data.AUTOTUNE))
    return train_dataset, test_dataset


def create_encoder(train_dataset, vocab_size):
    print(f"Tworzenie encodera z vocab_size={vocab_size}...")
    encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
    encoder.adapt(train_dataset.map(lambda text, label: text))
    return encoder



def build_model(encoder, config):

    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=config.EMBEDDING_DIM,
            mask_zero=True
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(config.LSTM_UNITS, return_sequences=True)
        ),
        tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(config.LSTM_UNITS // 2)
        ),
        tf.keras.layers.Dense(config.DENSE_UNITS, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1)
    ])
    return model


def compile_model(model, config):
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        optimizer=tf.keras.optimizers.Adam(config.LEARNING_RATE),
        metrics=['accuracy']
    )
    return model



def train_model(model, train_dataset, test_dataset, config):
    print(f"Rozpoczynam trening przez {config.EPOCHS} epok...")
    history = model.fit(
        train_dataset,
        epochs=config.EPOCHS,
        validation_data=test_dataset,
        validation_steps=30
    )
    return history


def evaluate_model(model, test_dataset):
    print("\nEwaluacja modelu...")
    test_loss, test_acc = model.evaluate(test_dataset)
    print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    return test_loss, test_acc


def test_predictions(model):
    sample_positive = ('The movie was cool. The animation and the graphics '
                       'were out of this world. I would recommend this movie.')
    sample_negative = ('The movie awful. Waste of time and money. '
                       'The director should consider different career path.')

    predictions = model.predict(
        tf.constant([sample_positive, sample_negative], dtype=tf.string)
    )

    print("\n=== Testowe predykcje ===")
    print(f"Pozytywny tekst: {predictions[0][0]:.4f} "
          f"({'POZYTYWNY' if predictions[0][0] > 0 else 'NEGATYWNY'})")
    print(f"Negatywny tekst: {predictions[1][0]:.4f} "
          f"({'POZYTYWNY' if predictions[1][0] > 0 else 'NEGATYWNY'})")



def save_model(model, path):
    print(f"\nZapisuję model do: {path}")
    model.save(path)
    print("Model zapisany pomyślnie!")



def main():
    config = Config()

    # Ładowanie danych
    train_dataset, test_dataset, info = load_data()

    # Przykładowe dane
    print("\n=== Przykładowe dane ===")
    for example, label in train_dataset.take(1):
        print('text: ', example.numpy()[:100], '...')
        print('label: ', label.numpy())

    # Przygotowanie datasetów
    train_dataset, test_dataset = prepare_datasets(
        train_dataset, test_dataset, config
    )

    # Tworzenie encodera
    encoder = create_encoder(train_dataset, config.VOCAB_SIZE)

    # Budowanie modelu
    print("\nBudowanie modelu...")
    model = build_model(encoder, config)
    model = compile_model(model, config)
    model.summary()

    # Trenowanie
    history = train_model(model, train_dataset, test_dataset, config)

    # Ewaluacja
    evaluate_model(model, test_dataset)

    # Testowe predykcje
    test_predictions(model)

    # Zapisywanie modelu
    save_model(model, config.MODEL_SAVE_PATH)

    # Wykresy
    plot_training_history(history)

    print("\n=== Trening zakończony ===")
    print(f"Model zapisany w: {config.MODEL_SAVE_PATH}")


if __name__ == '__main__':
    main()
