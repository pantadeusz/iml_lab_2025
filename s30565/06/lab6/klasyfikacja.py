import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds

print("Wczytywanie modelu z pliku 'best_beans_model.keras'...")
model = tf.keras.models.load_model('best_beans_model.keras')
print("Model wczytany.")


def preprocess_image(image, label):
    image = tf.image.resize(image, (128, 128))
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

ds_test = tfds.load('beans', split='test', as_supervised=True)
ds_test = ds_test.map(preprocess_image).batch(32).take(1)

for images, labels in ds_test:
    sample_image = images[0]
    true_label_index = labels[0].numpy()
    break

image_for_prediction = tf.expand_dims(sample_image, axis=0)

class_names = ['Angular Leaf Spot', 'Bean Rust', 'Healthy']
true_label_name = class_names[true_label_index]



predictions = model.predict(image_for_prediction)

predicted_class_index = np.argmax(predictions[0])
predicted_class_name = class_names[predicted_class_index]
confidence = np.max(predictions[0]) * 100

print("\n--- Wyniki klasyfikacji ---")
print(f"Prawdziwa etykieta: {true_label_name}")
print(f"Przewidziana etykieta: {predicted_class_name}")
print(f"Pewność (confidence): {confidence:.2f}%")

if predicted_class_name == true_label_name:
    print("\nPredykcja jest poprawna! :)")
else:
    print("\nPredykcja jest niepoprawna. :(")