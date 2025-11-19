import tensorflow as tf
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.layers.core import dropout
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras import regularizers
import pandas as pd

wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets
y = np.asarray(y).ravel().astype(int)
y = y - 1
y = to_categorical(y, num_classes=3)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=111)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
input = X_train.shape[1]

model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(input,)),
    tf.keras.layers.Dense(16, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(8, activation='relu',
                          kernel_regularizer=regularizers.l2(0.001)),
    tf.keras.layers.BatchNormalization(),
    # tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, validation_split=0.1, verbose=1)

pred_probs = model.predict(X_test)
y_pred = np.argmax(pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, y_pred, target_names=['Class 1', 'Class 2', 'Class 3']))