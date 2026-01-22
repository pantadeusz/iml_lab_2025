import tensorflow as tf
from base_model import load_mnist_data, evaluate_model


data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomRotation(0.03),
    tf.keras.layers.RandomTranslation(height_factor=0.1, width_factor=0.05),
    tf.keras.layers.RandomInvert(factor=0.04)
])

def augment_func(image, label):
    image = data_augmentation(image)
    return image, label

def load_model(path):
    model = tf.keras.models.load_model(path)
    return model

def main():
    model = load_model('MNIST_cnn_model.keras')
    ds_train, ds_test, ds_info = load_mnist_data(batch_size=54,augment=True,augment_func=augment_func)
    evaluate_model(model, ds_test)

if __name__ == '__main__':
    main()