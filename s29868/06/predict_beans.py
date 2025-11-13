import sys
from tensorflow import keras
import numpy as np
from PIL import Image
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # logs off

CLASS_NAMES = {
    0: 'angular_leaf_spot',
    1: 'bean_rust',
    2: 'healthy'
}


def preprocess_image(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def predict_image(model_path, img_path):
    model = keras.models.load_model(model_path, compile=False)

    img_array = preprocess_image(img_path)

    predictions = model.predict(img_array)
    class_id = np.argmax(predictions)
    class_name = CLASS_NAMES[class_id]
    confidence = predictions[0][class_id]
    print(f"Predicted class: {class_name} ({confidence * 100:.2f}%)")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python predict_beans.py <model_path> <image_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    img_path = sys.argv[2]
    predict_image(model_path, img_path)