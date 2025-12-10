from keras.models import load_model
import numpy as np
from PIL import Image
import sys

model = load_model("beans_best_model.keras")

class_names = ["angular_leaf_spot", "bean_rust", "healthy"]

def prepare_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(image_path):
    img_array = prepare_image(image_path)
    preds = model.predict(img_array)
    class_index = np.argmax(preds, axis=1)[0]
    class_name = class_names[class_index]
    confidence = preds[0][class_index]
    print(f"Klasa: {class_name}, Pewność: {confidence:.2f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Użycie: python classify_bean.py <ścieżka_do_obrazu>")
        sys.exit(1)

    image_path = sys.argv[1]
    classify_image(image_path)