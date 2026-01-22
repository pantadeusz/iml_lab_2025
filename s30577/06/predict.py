import os
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model_beans.keras')
classes = ['angular_leaf_spot', 'bean_rust', 'healthy']

for filename in os.listdir('photo'):
    path = os.path.join('photo', filename)
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, [128, 128]) / 255.0
    img = tf.expand_dims(img, 0)

    pred = model.predict(img, verbose=0)[0]
    print(f"{filename}: {classes[np.argmax(pred)]} ({np.max(pred) * 100:.1f}%)")