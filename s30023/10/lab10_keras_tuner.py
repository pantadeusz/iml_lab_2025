import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_tuner as kt

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

def create_model(hp, encoder):
    lstm_units_1 = hp.Int('lstm_units_1', min_value=32, max_value=128, step=32)
    lstm_units_2 = hp.Int('lstm_units_2', min_value=16, max_value=64, step=16)
    dense_units = hp.Int('dense_units', min_value=32, max_value=128, step=16)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    drop_amount = hp.Choice('drop_amount', values=[0.2, 0.3, 0.4, 0.5])


    model = tf.keras.Sequential([
        encoder,
        tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_1, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_units_2)),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dropout(drop_amount),
        tf.keras.layers.Dense(1)
    ])

    model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
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
    plt.savefig('history_plot.png')
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

    tuner = kt.Hyperband(
        lambda hp: create_model(hp, encoder),
        objective='val_accuracy',
        max_epochs=6,
        factor=3,
        directory='lab10_keras_tuner',
        project_name='lab10'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)

    tuner.search(train_dataset,
                 epochs=10,
                 validation_data=test_dataset,
                 callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(f"""
            Best HPs:
            LSTM 1 units: {best_hps.get('lstm_units_1')}
            LSTM 2 units: {best_hps.get('lstm_units_2')}
            Dense units: {best_hps.get('dense_units')}
            Learning rate: {best_hps.get('learning_rate')}
            Drop amount: {best_hps.get('drop_amount')}
    """)

    model = tuner.hypermodel.build(best_hps)

    history = model.fit(train_dataset, epochs=10, validation_data=test_dataset)

    plot_history(history)
    evaluate_model(model, test_dataset)
    sample_model_predict(model)
    save_model_to_disk(model)

if __name__ == '__main__':
    main()
