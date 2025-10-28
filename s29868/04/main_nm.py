import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

mushroom = fetch_ucirepo(id=73)

X = mushroom.data.features
y = mushroom.data.targets
X = X.apply(LabelEncoder().fit_transform)
y = LabelEncoder().fit_transform(y.values.ravel())
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = tf.keras.models.Sequential([
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10)
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)

print(classification_report(y_test, y_pred))

matrix = tf.math.confusion_matrix(y_test, y_pred)
matrix = matrix.numpy()
ConfusionMatrixDisplay(matrix).plot()
plt.show()

model.save('mmodel.h5')