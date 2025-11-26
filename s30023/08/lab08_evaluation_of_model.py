from lab08_baseline_model import evaluate_model, load_data
import tensorflow as tf

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.05),
    tf.keras.layers.RandomTranslation(height_factor=0.08, width_factor=0.08),
    tf.keras.layers.RandomInvert(factor=0.2)
])

def load_saved_model(path='tensorflow_MNIST_base.keras'):
    model = tf.keras.models.load_model(path)
    return model

def augment_image(image, label):
    image = data_augmentation(image)
    return image, label

def main():
    model = load_saved_model()
    ds_train, ds_test, ds_info = load_data()

    ds_test = ds_test.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)

    evaluate_model(model, ds_test)

if __name__ == '__main__':
    main()