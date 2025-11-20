import tensorflow as tf
from tensorflow import keras
import numpy as np
import sys
import os

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]

def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.resize(image, [128, 128])
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, axis=0)
    return image

def predict_image(model_path, image_path):
    if not os.path.exists(image_path):
        return

    model = keras.models.load_model(model_path)
    print(f"Załadowano model: {model_path}")

    image = preprocess_image(image_path)

    preds = model.predict(image)
    pred_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    print(f"Obraz: {image_path}")
    print(f"Przewidywana klasa: {CLASS_NAMES[pred_class]}")
    print(f"Pewność modelu: {confidence*100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(1)

    image_path = sys.argv[1]
    predict_image("beans_model_final.h5", image_path)
