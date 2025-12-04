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
    BATCH_SIZE = 64
    train_ds = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    return train_ds, test_ds

def create_text_encoder(train_ds):
    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
    encoder.adapt(train_ds.map(lambda text, label: text))
    return encoder

def create_model(encoder):
    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(
            input_dim=len(encoder.get_vocabulary()),
            output_dim=64,
            # Use masking to handle the variable sequence lengths
            mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    return model

def run_training(model, train_ds, test_ds):
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), 
                  optimizer=tf.keras.optimizers.Adam(1e-4), 
                  metrics=['accuracy'])
    history = model.fit(train_ds,
                        epochs=10,
                        validation_data=test_ds,
                        validation_steps=30)
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