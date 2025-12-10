from model_evaluation import augment_func
from base_model import evaluate_model, save_model
from tensorflow import keras
from base_model import load_mnist_data
from keras import layers



def create_model(input_shape=(28, 28, 1), num_of_classes=10):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),

        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_of_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    return model

def main():
    ds_train, ds_test, ds_info = load_mnist_data(batch_size=54, augment=True, augment_func=augment_func)
    model = create_model()

    model.fit(
        ds_train,
        epochs=5,
        validation_data=ds_test
    )

    evaluate_model(model, ds_test)

    save_model(model, 'MNIST_cnn_model.keras')


if __name__ == '__main__':
    main()