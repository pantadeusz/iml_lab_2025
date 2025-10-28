import tensorflow as tf
from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

iris = load_iris(as_frame=True)
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.Input((4,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

predictions = model(X_train[:1]).numpy()
print(f'predictions =\n{predictions}')

print(f'tf.nn.softmax(predictions).numpy() =\n{tf.nn.softmax(predictions).numpy()}')

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

print(f'-tf.math.log(1/3)={-tf.math.log(1/3)}')
print(f'loss_fn(y_train[:1], predictions).numpy()=\n{loss_fn(y_train[:1], predictions).numpy()}')

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50)

print(f'---EVALUATION---\n={model.evaluate(X_test, y_test, verbose=2)}')

