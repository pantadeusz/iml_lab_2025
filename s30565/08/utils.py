import tensorflow as tf
import numpy as np


def normalize_img(image, label):
    image = tf.cast(image, tf.float32) / 255.
    # MNIST z tfds już ma kształt (28, 28, 1) - nie trzeba dodawać wymiaru kanału
    return image, label


def data_augmentation_fn(image, label, max_shift_pixels=2, rotation_factor=0.05):
    # Inwersja kolorów z 50% szansą
    image = tf.cond(
        tf.random.uniform(()) > 0.5,
        lambda: 1.0 - image,
        lambda: image
    )

    # Rotacja - używamy tf.image z interpolacją
    random_angle = tf.random.uniform(
        shape=[],
        minval=-rotation_factor * 2 * 3.14159,  # konwersja na radiany
        maxval=rotation_factor * 2 * 3.14159
    )
    # tfa.image.rotate lub własna implementacja przez transform
    image = rotate_image(image, random_angle)

    # Translacja (przesunięcie)
    max_shift_factor = max_shift_pixels / 28.0
    shift_y = tf.random.uniform(shape=[], minval=-max_shift_factor, maxval=max_shift_factor)
    shift_x = tf.random.uniform(shape=[], minval=-max_shift_factor, maxval=max_shift_factor)
    image = translate_image(image, shift_x, shift_y)

    return image, label


def rotate_image(image, angle):
    """Rotacja obrazu o zadany kąt (w radianach)."""
    cos_angle = tf.cos(angle)
    sin_angle = tf.sin(angle)

    # Macierz transformacji afinicznej dla rotacji wokół środka
    # [cos, -sin, tx]
    # [sin,  cos, ty]
    # gdzie tx, ty kompensują przesunięcie środka
    transform = [cos_angle, -sin_angle, 0.0, sin_angle, cos_angle, 0.0, 0.0, 0.0]

    image_4d = tf.expand_dims(image, 0)
    rotated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image_4d,
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.shape(image)[:2],
        fill_value=0.0,
        interpolation='BILINEAR'
    )
    return tf.squeeze(rotated, 0)


def translate_image(image, shift_x, shift_y):
    """Translacja obrazu o zadane przesunięcie (w ułamku rozmiaru)."""
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)

    # Przesunięcie w pikselach
    tx = shift_x * w
    ty = shift_y * h

    # Macierz transformacji: [1, 0, tx, 0, 1, ty, 0, 0]
    transform = [1.0, 0.0, tx, 0.0, 1.0, ty, 0.0, 0.0]

    image_4d = tf.expand_dims(image, 0)
    translated = tf.raw_ops.ImageProjectiveTransformV3(
        images=image_4d,
        transforms=tf.expand_dims(transform, 0),
        output_shape=tf.shape(image)[:2],
        fill_value=0.0,
        interpolation='BILINEAR'
    )
    return tf.squeeze(translated, 0)


def evaluate_model_metrics(model, dataset, name="Model"):
    print(f"\n--- Ewaluacja: {name} ---")
    results = model.evaluate(dataset, verbose=0)

    metrics = {
        'Loss': results[0],
        'Accuracy': results[1]
    }
    print(f"Loss: {metrics['Loss']:.4f}, Accuracy: {metrics['Accuracy']:.4f}")
    return metrics