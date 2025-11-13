import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import sys
from tensorflow.keras.preprocessing import image

IMG_SIZE = 100
MODEL_PATH = "tensorflow_model_beans.keras"

def classify_dataset(model, ds_test, target_names):
    y_pred_prob = model.predict(ds_test)
    y_pred = y_pred_prob.argmax(axis=1)
    y_true = np.concatenate([y for x, y in ds_test], axis=0)

    print("\n--- SAMPLE RESULTS ---")
    for i in range(5):
        print(f"Obraz {i+1}: Prawdziwa klasa = {target_names[y_true[i]]}, "
              f"Przewidziana = {target_names[y_pred[i]]}")


def classify_single_image(model, image_path, target_names):
    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    print(f"\n--- PREDICTION ---")
    print(f"Plik: {image_path}")
    print(f"Przewidziana klasa: {target_names[predicted_class]}")
    print(f"Pewność: {confidence:.2f}")


def main():
    print(f"Ładowanie modelu z pliku: {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH)

    if len(sys.argv) > 1:
        _, ds_info = tfds.load('beans', split=['test'], with_info=True, as_supervised=True)
        target_names = ds_info.features['label'].names
        image_path = sys.argv[1]
        classify_single_image(model, image_path, target_names)
    else:
        print("Give me link for image in format [python load_model.py img.png]")


if __name__ == "__main__":
    main()