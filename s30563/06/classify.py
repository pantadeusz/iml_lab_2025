import sys
import tensorflow as tf
from tensorflow import keras

def classify_image(model_path, img_path, img_size=(128, 128)):
    model = keras.models.load_model(model_path)

    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.expand_dims(image, 0)

    preds = model.predict(image)
    predicted_class = tf.argmax(preds, axis=1).numpy()[0]

    print(f"Przewidziana klasa: {predicted_class}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("python classify_beans.py <ścieżka_do_modelu> <ścieżka_do_obrazu>")
        sys.exit(1)

    model_path = sys.argv[1]
    image_path = sys.argv[2]
    classify_image(model_path, image_path)