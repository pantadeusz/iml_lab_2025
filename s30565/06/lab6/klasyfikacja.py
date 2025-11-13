import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import os

image_path = ''

if not os.path.exists(image_path):
    print("Nie znaleziono obrazu!")
    exit()

model = tf.keras.models.load_model('best_beans_model.keras')

def load_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=(128, 128), color_mode='rgb')
    img_array = tf.keras.utils.img_to_array(img)
    img_array = img_array / 255.0
    img_array = tf.expand_dims(img_array, axis=0)
    return img_array

image_for_prediction = load_image(image_path)

class_names = ['Angular Leaf Spot (Plamistość)', 'Bean Rust (Rdza)', 'Healthy (Zdrowa)']
predictions = model.predict(image_for_prediction)
predicted_class_index = np.argmax(predictions[0])
predicted_class_name = class_names[predicted_class_index]
confidence = np.max(predictions[0]) * 100

print("\n--- Wynik diagnozy ---")
print(f"Diagnoza: {predicted_class_name}")
print(f"Pewność modelu: {confidence:.2f}%")
print(f"Surowe wyniki (prawdopodobieństwa): {predictions[0]}")