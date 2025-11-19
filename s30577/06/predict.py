import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path


CLASS_NAMES = ['angular_leaf_spot', 'bean_rust', 'healthy']


def load_and_preprocess_image(image_path):
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Nie znaleziono pliku: {image_path}")
    
    image = tf.io.read_file(image_path)
    
    image = tf.image.decode_image(image, channels=3, expand_animations=False)

    image = tf.image.resize(image, [128, 128])
    
    image = tf.cast(image, tf.float32) / 255.0
    
    image = tf.expand_dims(image, 0)  
    
    return image


def predict_image(model_path, image_path):

    model = keras.models.load_model(model_path)
    image = load_and_preprocess_image(image_path)
    predictions = model.predict(image, verbose=0)[0]
    
    predicted_class_idx = np.argmax(predictions)
    predicted_class_name = CLASS_NAMES[predicted_class_idx]
    confidence = predictions[predicted_class_idx] * 100
    
    print(f"\nObraz: {Path(image_path).name}")
    print(f"Przewidziana klasa: {predicted_class_name}")
    print(f"Pewność: {confidence:.2f}%\n")
    
    return predicted_class_name, confidence


def main():

    model_path = 'beans_best_model.keras'
    
    if len(sys.argv) < 2:
        print("UŻYCIE:")
        print(f"  python {sys.argv[0]} <ścieżka_do_obrazu>")
        print("\nPRZYKŁAD:")
        print(f"  python {sys.argv[0]} moj_lisek.jpg")
        print(f"  python {sys.argv[0]} ~/Downloads/bean_leaf.png")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        predict_image(model_path, image_path)
    except FileNotFoundError as e:
        print(f"BŁĄD: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"BŁĄD podczas przetwarzania: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()