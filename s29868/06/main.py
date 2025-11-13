import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import matplotlib.pyplot as plt
from kerastuner.tuners import RandomSearch

BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)
NUM_CLASSES = 3

def load_datasets():
    ds_train, ds_val, ds_test = tfds.load(
        'beans',
        split=['train', 'validation', 'test'],
        shuffle_files=True,
        as_supervised=True
    )
    return ds_train, ds_val, ds_test

def preprocessing(image, label):
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def prepare_datasets(ds_train, ds_val, ds_test):
    ds_train = ds_train.map(preprocessing).batch(BATCH_SIZE)
    ds_val = ds_val.map(preprocessing).batch(BATCH_SIZE)
    ds_test = ds_test.map(preprocessing).batch(BATCH_SIZE)
    return ds_train, ds_val, ds_test

def create_mlp_model(initializer='glorot_uniform', activation='relu', optimizer='adam'):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(128, 128, 3)),
        keras.layers.Dense(128, activation=activation, kernel_initializer=initializer),
        keras.layers.Dense(64, activation=activation, kernel_initializer=initializer),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_model(model, ds_train, ds_val, epochs=25):
    history = model.fit(ds_train, validation_data=ds_val, epochs=epochs)
    return history

def evaluate_model(model, ds_test):
    loss, acc = model.evaluate(ds_test)
    print('Test accuracy:', acc)
    return loss, acc

def plot_history(history):
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

def build_tuned_model(hp):
    model = keras.Sequential()
    model.add(keras.Input(shape=(128, 128, 3)))
    model.add(keras.layers.Flatten())
    model.add(
        keras.layers.Dense(
            units=hp.Int('units1', min_value=32, max_value=256, step=32),
            activation=hp.Choice('activation1', ['relu', 'tanh', 'sigmoid']),
            kernel_initializer=hp.Choice('init1', ['glorot_uniform', 'he_normal'])
        )
    )
    model.add(
        keras.layers.Dense(
            units=hp.Int('units2', min_value=32, max_value=128, step=32),
            activation=hp.Choice('activation2', ['relu', 'tanh', 'sigmoid']),
            kernel_initializer=hp.Choice('init2', ['glorot_uniform', 'he_normal'])
        )
    )
    model.add(keras.layers.Dense(NUM_CLASSES, activation='softmax'))
    optimizer_choice = hp.Choice('optimizer', ['adam', 'sgd', 'rmsprop'])
    if optimizer_choice == 'adam':
        optimizer = keras.optimizers.Adam()
    elif optimizer_choice == 'sgd':
        optimizer = keras.optimizers.SGD()
    else:
        optimizer = keras.optimizers.RMSprop()
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def create_cnn_model():
    model = keras.Sequential([
        keras.layers.Input(shape=(128,128,3)),
        keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2,2)), # Downsamples the input along its spatial dimensions (height and width) by
        # taking the maximum value over an input window
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.25),
        keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D((2,2)),
        keras.layers.Dropout(0.25),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    ds_train, ds_val, ds_test = load_datasets()
    ds_train, ds_val, ds_test = prepare_datasets(ds_train, ds_val, ds_test)

    model = create_mlp_model()
    history = train_model(model, ds_train, ds_val)
    evaluate_model(model, ds_test)
    plot_history(history)
    model.save('default_model.keras')

    tuner = RandomSearch(
        build_tuned_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=1,
        directory='tuner_logs',
        project_name='beans_mlp'
    )
    tuner.search(ds_train, validation_data=ds_val, epochs=10)
    best_model = tuner.get_best_models(num_models=1)[0]
    best_model.save('tuned_model.keras')

    cnn_model = create_cnn_model()
    history_cnn = train_model(cnn_model, ds_train, ds_val)
    evaluate_model(cnn_model, ds_test)
    plot_history(history_cnn)
    cnn_model.save('cnn_model.keras')

if __name__ == "__main__":
    main()