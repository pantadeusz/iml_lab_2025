import keras
import tensorflow as tf
import tensorflow_datasets as tfds
import keras_tuner as kt

from keras import Input


tfds.disable_progress_bar()
MODEL_FILENAME = 'sentiment_model.keras'
TUNER_DIR = 'kt_tuning'


class SentimentModelTrainer:
    def __init__(self, vocab_size=10000, buffer_size=100000, batch_size=64,
                 dataset_name='imdb_reviews'):
        self.model = None
        self.test_dataset = None
        self.train_dataset = None
        self.encoder = None
        self.VOCAB_SIZE = vocab_size
        self.BUFFER_SIZE = buffer_size
        self.BATCH_SIZE = batch_size
        self.dataset_name = dataset_name

    def load_and_prepare_data(self):
        print("Loading and Preparing Data...")
        dataset, info = tfds.load(self.dataset_name, as_supervised=True, with_info=True)
        train_ds, test_ds = dataset['train'], dataset['test']

        self.train_dataset = train_ds.shuffle(self.BUFFER_SIZE).batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        self.test_dataset = test_ds.batch(self.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

        self.encoder = tf.keras.layers.TextVectorization(
            max_tokens=self.VOCAB_SIZE
        )
        self.encoder.adapt(self.train_dataset.map(lambda text, label: text))

        print(f"Data Loaded! Dictionary size: {len(self.encoder.get_vocabulary())}")

    def build_model(self, hp):
        embed_dim = hp.Int('embedding_dim', min_value=32, max_value=128, step=32)
        gru_units = hp.Int('gru_units', min_value=32, max_value=128, step=32)
        dense_units = hp.Int('dense_units', min_value=32, max_value=64, step=32)
        dropout_rate = hp.Float('dropout_rate', min_value=0.0, max_value=0.5, step=0.1)
        learning_rate = hp.Choice('learning_rate', values=[1e-4, 5e-4, 1e-3])

        model = keras.Sequential([
            self.encoder,
            tf.keras.layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=embed_dim,
                mask_zero=True),
            tf.keras.layers.GRU(gru_units, return_sequences=True),
            tf.keras.layers.GRU(gru_units // 2),
            tf.keras.layers.Dense(dense_units, activation='relu'),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1)
        ])

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      metrics=['accuracy'])
        return model

    def tune_and_train_model(self, epochs=3):
        if self.encoder is None:
            raise RuntimeError('Encoder must be adapted before tuning.')

        print("\nTurning on Keras Tuner...")

        tuner = kt.Hyperband(
            self.build_model,
            objective='val_accuracy',
            max_epochs=epochs,
            factor=3,
            directory=TUNER_DIR,
            project_name='sentiment_tuning',
            overwrite=True
        )

        tuner.search(self.train_dataset, epochs=epochs, validation_data=self.test_dataset)

        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = tuner.hypermodel.build(best_hps)

        print("\nFound the best hyperparameters:")
        print(best_hps.values)

        print("\nTraining the final model...")
        history = self.model.fit(self.train_dataset, epochs=epochs,
                                 validation_data=self.test_dataset)

        test_loss, test_acc = self.model.evaluate(self.test_dataset)
        print(f'\n--- Final Results ---')
        print(f'Test Loss: {test_loss:.4f}')
        print(f'Test Accuracy: {test_acc:.4f}')
        print('------------------------')
        return history

    def save_model(self):
        model_export = tf.keras.Sequential([
            Input(shape=(1,), dtype=tf.string),
            self.encoder,
            *self.model.layers[1:]
        ])

        model_export.compile(
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            optimizer=tf.keras.optimizers.Adam(),
            metrics=['accuracy']
        )

        model_export.save(MODEL_FILENAME, save_format='keras')
        print(f"Model successfully saved as {MODEL_FILENAME}!")


if __name__ == '__main__':
    try:
        trainer = SentimentModelTrainer(vocab_size=10000)
        trainer.load_and_prepare_data()
        trainer.tune_and_train_model(epochs=3)
        trainer.save_model()
    except Exception as e:
        print(f"An Error occurred during the process: {e}")