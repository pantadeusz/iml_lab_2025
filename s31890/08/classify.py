import numpy as np
from PIL import Image
import argparse
import keras


def load_and_preprocess_image(image_path, target_size=(500, 500)):
    img = Image.open(image_path)

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = img.resize(target_size)

    img_array = np.array(img, dtype=np.uint8)

    img_array = np.expand_dims(img_array, axis=0)

    return img_array

def classify_image(model_path, image_path, class_names=None):
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully.")
    if not model:
        raise ValueError(f"Model on path {model_path} failed to load or is missing")
    
    print(f"Loading and preprocessing image from {image_path}...")
    img_array = load_and_preprocess_image(image_path)
    print(f"Image shape after preprocessing: {img_array.shape}")
    
    print("Running inference...")
    predictions = model.predict(img_array, verbose=0)
    
    predicted_class_index = np.argmax(predictions[0])
    predicted_class_confidence = predictions[0][predicted_class_index]
    
    if class_names:
        predicted_class_name = class_names[predicted_class_index]
    else:
        predicted_class_name = f"Class {predicted_class_index}"
    
    print(f"Predicted class: {predicted_class_name} (index: {predicted_class_index})")
    print(f"Confidence: {predicted_class_confidence:.4f}")
    
    return predicted_class_index, predicted_class_name, predicted_class_confidence

def main():
    parser = argparse.ArgumentParser(description="Classify an image using a trained Keras model.")
    parser.add_argument("image_path", help="Path to the image file to classify")
    parser.add_argument("--model", default="best_tuner_model.keras", help="Path to the saved model file")
    parser.add_argument("--classes", nargs="*", help="List of class names (optional)")
    
    args = parser.parse_args()
    
    default_class_names = ['angular_leaf_spot', 'bean_rust', 'healthy']
    
    class_names = args.classes or default_class_names
    
    result = classify_image(
        model_path=args.model,
        image_path=args.image_path,
        class_names=class_names
    )
    
    print(f"\nFinal Prediction: {result[1]} (Confidence: {result[2]:.4f})")

if __name__ == "__main__":
    main()

