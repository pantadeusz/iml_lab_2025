import tensorflow as tf
import tensorflow_datasets as tfds
import os

BUFFER_SIZE = 10000
BATCH_SIZE = 64
VOCAB_SIZE = 1000
EPOCHS = 3
MODEL_FILENAME = 'model_sentymentu.keras'


class SentimentTrainer:
    def __init__(self):
        self.model = None
        self.encoder = None
        self.train_dataset = None
        self.test_dataset = None

    def load_and_prepare_data(self):
        dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
        train_ds, test_ds = dataset['train'], dataset['test']

        self.train_dataset = train_ds.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        self.test_dataset = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    def build_encoder(self):
        self.encoder = tf.keras.layers.TextVectorization(max_tokens=VOCAB_SIZE)
        self.encoder.adapt(self.train_dataset.map(lambda text, label: text))

    def build_model(self):
        self.model = tf.keras.Sequential([
            self.encoder,
            tf.keras.layers.Embedding(
                input_dim=len(self.encoder.get_vocabulary()),
                output_dim=64,
                mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                           optimizer=tf.keras.optimizers.Adam(1e-4),
                           metrics=['accuracy'])

    def train(self):
        print(f"Trening ({EPOCHS} epoki) ---")
        history = self.model.fit(
            self.train_dataset,
            epochs=EPOCHS,
            validation_data=self.test_dataset,
            validation_steps=30
        )
        return history

    def save(self, path):
        self.model.save(path)

if __name__ == "__main__":
    trainer = SentimentTrainer()
    trainer.load_and_prepare_data()
    trainer.build_encoder()
    trainer.build_model()
    trainer.train()
    trainer.save(MODEL_FILENAME)