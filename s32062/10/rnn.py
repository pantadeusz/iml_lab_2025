#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

import tensorflow_datasets as tfds
import tensorflow as tf

tfds.disable_progress_bar()

import matplotlib.pyplot as plt

import sys


def plot_graphs(history, metric):
  plt.plot(history.history[metric])
  plt.plot(history.history['val_'+metric], '')
  plt.xlabel("Epochs")
  plt.ylabel(metric)
  plt.legend([metric, 'val_'+metric])

def prepare_data():
    dataset, info = tfds.load('imdb_reviews', with_info=True,
                              as_supervised=True)
    train_dataset, test_dataset = dataset['train'], dataset['test']

    BUFFER_SIZE = 10000
    BATCH_SIZE = 64

    train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    return train_dataset, test_dataset

def create_encoder(train_dataset):
    VOCAB_SIZE = 1000
    encoder = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE)
    encoder.adapt(train_dataset.map(lambda text, label: text))

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


def main():

    train_dataset, test_dataset = prepare_data()
    encoder = create_encoder(train_dataset)
    model = create_model(encoder)
    # sample_text = ('The movie was cool. The animation and the graphics '
    #                'were out of this world. I would recommend this movie.')
    while True:
        print("Podaj tekst:\n")
        for line in sys.stdin:
            predictions = model.predict(tf.constant([line], dtype=tf.string))

            print(predictions[0])

    # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               optimizer=tf.keras.optimizers.Adam(1e-4),
    #               metrics=['accuracy'])
    #
    # history = model.fit(train_dataset, epochs=10,
    #                     validation_data=test_dataset,
    #                     validation_steps=30)
    #
    # test_loss, test_acc = model.evaluate(test_dataset)
    #
    # print('Test Loss:', test_loss)
    # print('Test Accuracy:', test_acc)
    #
    # plt.figure(figsize=(16, 8))
    # plt.subplot(1, 2, 1)
    # plot_graphs(history, 'accuracy')
    # plt.ylim(None, 1)
    # plt.subplot(1, 2, 2)
    # plot_graphs(history, 'loss')
    # plt.ylim(0, None)

    # sample_positive_text = ('The movie was cool. The animation and the graphics '
    #                         'were out of this world. I would recommend this movie.')
    # sample_negative_text = ('The movie awful. Waste of time and money.'
    #                         'The director should consider different career path.')
    # predictions = model.predict(tf.constant([sample_positive_text, sample_negative_text], dtype=tf.string))
    # print(predictions)
    #
    # model = tf.keras.Sequential([
    #     encoder,
    #     tf.keras.layers.Embedding(len(encoder.get_vocabulary()), 64, mask_zero=True),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    #     tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(1)
    # ])
    #
    # model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    #               optimizer=tf.keras.optimizers.Adam(1e-4),
    #               metrics=['accuracy'])
    #
    # history = model.fit(train_dataset, epochs=10,
    #                     validation_data=test_dataset,
    #                     validation_steps=30)
    #
    # test_loss, test_acc = model.evaluate(test_dataset)
    #
    # print('Test Loss:', test_loss)
    # print('Test Accuracy:', test_acc)
    #
    # # predict on a sample text without padding.
    #
    # sample_text = ('The movie was not good. The animation and the graphics '
    #                'were terrible. I would not recommend this movie.')
    # predictions = model.predict(np.array([sample_text]))
    # print(predictions)
    #
    # plt.figure(figsize=(16, 6))
    # plt.subplot(1, 2, 1)
    # plot_graphs(history, 'accuracy')
    # plt.subplot(1, 2, 2)
    # plot_graphs(history, 'loss')


if __name__ == "__main__":
    main()

