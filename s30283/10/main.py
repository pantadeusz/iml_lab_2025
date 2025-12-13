import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

tfds.disable_progress_bar()

BATCH_SIZE = 128
EPOCHS = 20

def plot_graphs(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Batch size = {BATCH_SIZE}, epochs = {EPOCHS}')

    ax1.plot(history.history['accuracy'], color='c', label='accuracy')
    ax1.plot(history.history['val_accuracy'], color='darkorchid', label='val_accuracy')
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training and Validation Accuracy")
    ax1.legend()

    ax2.plot(history.history['loss'], color='c', label='loss')
    ax2.plot(history.history['val_loss'], color='darkorchid', label='val_loss')
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel("Loss")
    ax2.set_title("Training and Validation Loss")
    ax2.legend()

    plt.tight_layout()
    plt.savefig('graphs.png')
    print('Graphs generated and saved to graphs.png')

def load_datasets():
    dataset, info = tfds.load('imdb_reviews', with_info=True,
                            as_supervised=True)
    train_ds, test_ds = dataset['train'], dataset['test']
    return train_ds, test_ds

def prepare_datasets(train_ds, test_ds):
    BUFFER_SIZE = 10000
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
                        epochs=EPOCHS,
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
    plot_graphs(history)

if __name__ == '__main__':
    main()