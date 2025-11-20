import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import sys

IMAGE_SIZE = (128, 128)
MODEL_PATH = "best_beans_model.keras"
CLASS_NAMES = ["healthy", "angular leaf spot", "bean rust"]


def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB").resize(IMAGE_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(img_array, axis=0)


def classify_image(model_path, image_path):
    model = keras.models.load_model(model_path)
    img = preprocess_image(image_path)
    pred = model.predict(img)
    class_idx = np.argmax(pred)
    print(f"Predicted class: {CLASS_NAMES[class_idx]}")
    print(f"Probabilities: {pred[0]}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    classify_image(MODEL_PATH, image_path)
