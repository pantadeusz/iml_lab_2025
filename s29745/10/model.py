import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt

tfds.disable_progress_bar()

def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])

def load_and_prep_data(buffer_size=10000, batch_size=64):
    dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']
    train_dataset = train_dataset.shuffle(buffer_size).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, test_dataset

def create_encoder(train_dataset, vocab_size=1000):
    encoder = tf.keras.layers.TextVectorization(max_tokens=vocab_size)
    encoder.adapt(train_dataset.map(lambda text, label: text))
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

def predict_sentiment(model, text):
    tensor_text = tf.constant([text])
    prediction = model.predict(tensor_text)[0][0]
    print(f"Text: {text}")
    print(f"Logits: {prediction}")
    if prediction > 0:
        print("Positive")
    else:
        print("Negative")

def main():
    train_ds, test_ds = load_and_prep_data()
    encoder = create_encoder(train_ds)
    
    model = build_model(encoder)
    
    history = model.fit(train_ds, epochs=10, validation_data=test_ds, validation_steps=30)
    
    test_loss, test_acc = model.evaluate(test_ds)
    print(f'Test Loss: {test_loss}')
    print(f'Test Accuracy: {test_acc}')
    
    plt.figure(figsize=(16, 6))
    plt.subplot(1, 2, 1)
    plot_graphs(history, 'accuracy')
    plt.subplot(1, 2, 2)
    plot_graphs(history, 'loss')
    
    sample_text = ('The movie was not good. The animation and the graphics '
                   'were terrible. I would not recommend this movie.')
    predict_sentiment(model, sample_text)
    
    model.save('model.keras')

if __name__ == "__main__":
    main()