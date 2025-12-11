import tensorflow as tf
import argparse
import os
import numpy as np

CLASS_NAMES = ["angular_leaf_spot", "bean_rust", "healthy"]

def load_and_preprocess_image(filepath: str) -> tf.Tensor:
    img = tf.io.read_file(filepath)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (128, 128))
    img = tf.cast(img, tf.float32) / 255.0
    return img


def run_inference(image_path: str, model_path: str):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at this path: {image_path}")

    print("Loading model...")
    model = tf.keras.models.load_model(model_path)

    print("Loading and preprocessing image...")
    img = load_and_preprocess_image(image_path)
    img_batch = tf.expand_dims(img, axis=0)  #  we add batch since tf model was trained on batches (4dim shape) -> (batch_size, 128, 128, 3)

    print("Running inference...")
    preds = model.predict(img_batch)
    print(f'Preds: {preds}')
    predicted_index = np.argmax(preds)
    predicted_class = CLASS_NAMES[np.argmax(preds)]

    print(f"\nPredicted Class: {predicted_index} -> {predicted_class}")

    return predicted_class


def init_argparser():
    parser = argparse.ArgumentParser(
        prog='Model Inference',
        description='Test model inference on a provided image'
    )
    parser.add_argument(
        '-f', '--filepath',
        required=True,
        help="Path to image file"
    )
    parser.add_argument(
        '-m', '--model',
        default='best_model.keras',
        help="Path to the saved model"
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = init_argparser()
    run_inference(args.filepath, args.model)