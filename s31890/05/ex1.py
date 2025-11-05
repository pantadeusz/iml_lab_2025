import tensorflow as tf
import keras
import keras_tuner
from keras_tuner import HyperParameters as hp

print(f"TensorFlow version: {tf.__version__}")

mnist = keras.datasets.mnist

(X_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = X_train / 255.0, x_test / 255.0

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10),
    ]
)

predictions = model(x_train[:1]).numpy()
print(predictions)

tf.nn.softmax(predictions).numpy()

loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test, verbose=2)

model = keras.models.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        ),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10),
    ]
)
