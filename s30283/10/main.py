import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

tfds.disable_progress_bar()

def plot_graphs(history, metric):
    plt.plot(history.history[metric], color='c')
    plt.plot(history.history['val_'+metric], '', color='darkorchid')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

def load_datasets():
    dataset, info = tfds.load('imdb_reviews', with_info=True,
                            as_supervised=True)
    train_ds, test_ds = dataset['train'], dataset['test']
    return train_ds, test_ds

def prepare_datasets(train_ds, test_ds):
    BUFFER_SIZE = 10000
    BATCH_SIZE = 128
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds

def create_text_encoder(train_ds):
    '''
    Create text encoder layer that converts strings into sequences of integer token ids.
    hello world -> [12, 2, 1, 0, 2]
    '''

    # number of unique tokens
    VOCAB_SIZE = 1000

    # it will handle top 1000 (VOCAB_SIZE) most frequent tokens in trening, others will be oov -> out of vocabulary
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)

    # learns vocabulary from training set and transforms (text, label) pair into just text
    encoder.adapt(train_ds.map(lambda text, label: text))
    return encoder

def create_model(encoder):
    model = tf.keras.Sequential([
        encoder,

        # turn each token into 64-dim dense vector
        # embedded sequence = [[0.12, -0.4, ... (64 dims)], [...], [...]]
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths (ignore padding tokens -> 0)
            mask_zero=True),

        # recurrent layer long-short-term-memory layer
        # processes sequence forward and backward
        # learning context!
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),

        # learn nonlinear features from LSTM output
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

def run_training(model, train_ds, test_ds):
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                  optimizer=tf.keras.optimizers.Adam(1e-4), 
                  metrics=['accuracy'])
    history = model.fit(train_ds,
                        epochs=20,
                        validation_data=test_ds,
                        validation_steps=30)
    model.save('model.keras')
    model.summary()
    return model, history

def main():
    train_ds, test_ds = load_datasets()
    train_ds, test_ds = prepare_datasets(train_ds, test_ds)
    text_encoder = create_text_encoder(train_ds)
    model = create_model(text_encoder)
    model, history = run_training(model, train_ds, test_ds)
    plot_graphs(history, 'accuracy')

    test_loss, test_acc = model.evaluate(test_ds)

    print('Test Loss:', test_loss)
    print('Test Accuracy:', test_acc)
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.ylim(None, 1)
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    plt.ylim(0, None)

if __name__ == '__main__':
    main()


import tensorflow as tf
import argparse

def init_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argment('-p', '--path')
    return parser

def load_model(path: str):
    model = tf.load_model()
    return model

def read_input() -> str:
    user_input = input()
    return user_input.strip()

def analyze_sentiment(model, text: str) -> str:
    # preprocess text and load into model
    model()
    # return str 'positive' or 'negative' depending on model predictions
    result = ''
    return result

def main():
    parser = init_args()
    args = parser
    model_path = parser.args.path
    model = load_model(model_path)
    user_input = read_input()
    sentiment_answer = analyze_sentiment(model, user_input)
    print(f'Text is {sentiment_answer}')

if __name__ == '__main__':
    main()

## Wprowadzenie
Zadanie polega na analizie sentymentu tekstu z użyciem warstw rekurencyjnych w sieciach neuronowych.

## Zbiór danych
Dataset wykorzystany do eksperymetów - IMDB large movie review dataset. Jest to zbiór danych wykorzystywany do klasyfikacji binarnej. Polega na analizie sentymentu opinii użytkowników o filmach - pozytywna lub negatywna.

## Architektura modelu
| Wartstwa | Output Shape | Liczba Parametrów |
| :--- | :--- | :--- |
| Text Vectorization | (None, None) | 0 |
| Embedding | (None, None, 64) | 64 000 |
| Bidiretional LSTM | (None, 128) | 66 048 |
| Dense | (None, 64)  | 8 256 |
| Dense | (None, 1) | 65 |

Total params: 415,109 (1.58 MB)
Trainable params: 138,369 (540.50 KB)
Non-trainable params: 0 (0.00 B)
Optimizer params: 276,740 (1.06 MB)

## Trening
Model trenowany był na parę sposobów. Powiększyłem również liczbę epok i zmniejszyłem rozmiar batcha byśmy mogli dokładniej przypatrzeć się zmianie dokładności i stracie na przestrzeni czasu.

Początkowo w preprocessingu input jest przerabiany na sekwencje tokenów za pomocą warstwy TextVectorization.
Następnie przechodzi przez warstwę Embedding, która generuje embeddings dla każdego tokenu wymiaru 64.
Potem embeddingi przechodzą przez warstwę rekurencyjną czyli dwustronny LSTM, gdzie model uczy się zależności i kontekstu.
Na końcu znajduje się warstwa w pełni połączona, która tworzy 64 cechy, dzięki którym może dostrzegać różne relacje pomiędzy tokenami.

* Optymalizator - Adam, learning rate = 0.001
* Funkcja straty - binary crossentropy

### Przebiegi treningów dla poszczególnych ustawień
> `batch_size` = 64, `epochs` = 10

*Dokładność*

![](https://i.imgur.com/5GvlPYW.png)

*Strata*

![](https://i.imgur.com/QL5v0LE.png)

> `batch_size` = 128, `epochs` = 20

*Dokładność*

![](https://i.imgur.com/kQCByYX.png)

*Strata*

![](https://i.imgur.com/gy9kNyb.png)

## Dodatkowe funkcje
Stworzony został również plik `runner.py`, odpowiedzialny za przetestowanie działania wytrenowanego modelu. Wystarczy posłużyć się poniższą instrukcją, aby dokonać analizy sentymentu tekstu, który podamy w inpucie w konsoli:
```sh
python runner.py --path (ścieżka do modelu)
(Tutaj należy wprowadzić text)
>>> Wyświetli się odpowiedź modelu - pozytywna | negatywna
```

## Wnioski
