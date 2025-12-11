import tensorflow as tf
import keras_tuner as kt
from prompt_toolkit.input import Input
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

def load_data(random_state=42):
    iris = load_iris(as_frame=True)
    print(iris.data.columns)
    X = iris.data
    y = iris.target

    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=random_state)
    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(hp, input_shape=(4,), output_units=3):
    model = tf.keras.Sequential()

    model.add(tf.keras.Input(shape=input_shape))
    model.add(
        tf.keras.layers.Dense(
            units=hp.Int("units", min_value=32, max_value=512, step=32),
            activation="relu",
        )
    )
    model.add(tf.keras.layers.Dense(output_units, activation='softmax'))

    learning_rate = hp.Choice("lr", values=[0.001, 0.01, 0.1])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)

    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=["accuracy"],
    )

    # model = tf.keras.models.Sequential([
    #     tf.keras.Input(shape=input_shape),
    #     tf.keras.layers.Dense(hidden_units, activation='relu'),
    #     tf.keras.layers.Dense(output_units, activation='softmax')
    # ])

    # loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    # model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy'])

    return model

def evaluate_model(model, X_test, y_test):
    # results = model.evaluate(X_test, y_test, verbose=2)
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)

    print(
        f"""
        --- EVALUATION ---
        {classification_report(y_test, y_pred, target_names=['setosa', 'versicolor', 'virginica'])}
        """)


def save_model(model, path='tensorflow_model.keras'):
    model.save(path)
    print(f'Model saved to {path}')


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    tuner = kt.RandomSearch(
        hypermodel=build_model,
        objective="val_accuracy",
        max_trials=10,
        executions_per_trial=2,
        overwrite=True,
        directory="05",
        project_name="Lab5",
    )

    tuner.search_space_summary()
    tuner.search(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    best_hyper_param = tuner.get_best_hyperparameters(1)[0]
    model = build_model(best_hyper_param)

    model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

    print(
        f"""
        --- Best hyperparameters: --- 
        - Units in hidden layer: {best_hyper_param.get('units')}
        - Learning rate): {best_hyper_param.get('lr')}
        """
    )

    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()