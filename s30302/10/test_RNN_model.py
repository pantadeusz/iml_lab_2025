import sys
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('model_RNN.keras')

for line in sys.stdin:
    line = line.strip()
    if line:
        print("Wpisany tekst do predykcji")
        print(line)
        input_tensor = tf.constant([line], dtype=tf.string)
        prediction = model.predict(input_tensor)
        if prediction[0] > 0.5:
            print("Pozytywna ocena")
        else:
            print("Negatywna ocena")