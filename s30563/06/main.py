import keras
import tensorflow_datasets as tfds

def get_dataset():
    train_ds, val_ds, test_ds = tfds.load(
        'beans',
        split=['train', 'validation', 'test'],
        as_supervised=True
    )
    return train_ds, val_ds, test_ds

def create_model(input_shape, initializer='glorot_uniform', activation='relu', optimizer='adam'):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(128, activation=activation),
        keras.layers.Dense(64, activation=activation),
        keras.layers.Dense(32, activation=activation),
        keras.layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    pass

if __name__ == "__main__":
    main()

