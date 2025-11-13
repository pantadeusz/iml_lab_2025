import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np

def load_model_and_predict(model_path, image_index=0):
    model = keras.models.load_model(model_path)
    print(f"Model wczytany z: {model_path}")
    
    test_ds, info = tfds.load('beans', split='test', with_info=True, as_supervised=True)
    class_names = info.features['label'].names
    
    def preprocess(image, label):
        image = tf.image.resize(image, [128, 128])
        image = tf.cast(image, tf.float32) / 255.0
        return image, label
    
    test_ds = test_ds.map(preprocess)
    
    for i, (image, true_label) in enumerate(test_ds):
        if i == image_index:
            image_batch = tf.expand_dims(image, 0)
            
            predictions = model.predict(image_batch, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class] * 100
            
            print(f"\n=== Wynik klasyfikacji ===")
            print(f"Prawdziwa etykieta: {class_names[true_label.numpy()]}")
            print(f"Przewidziana klasa: {class_names[predicted_class]}")
            print(f"Pewność: {confidence:.2f}%")
            print(f"\nWszystkie prawdopodobieństwa:")
            for j, prob in enumerate(predictions[0]):
                print(f"  {class_names[j]}: {prob*100:.2f}%")
            break

if __name__ == "__main__":
    load_model_and_predict('beans_best_model.keras', image_index=5)