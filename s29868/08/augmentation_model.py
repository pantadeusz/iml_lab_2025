from base_model import load_mnist_data, evaluate_model, save_model, create_model
from model_evaluation import augment_func
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers, models
from tensorflow import keras
from sklearn.metrics import classification_report
import numpy as np

def main():
    ds_train, ds_test, ds_info = load_mnist_data(batch_size=54,augment=True,augment_func=augment_func)
    model = create_model()

    model.fit(
        ds_train,
        epochs=6,
        validation_data=ds_test,
    )

    evaluate_model(model, ds_test)

    save_model(model, path='MNIST_augmentation_model.keras')


if __name__ == '__main__':
    main()