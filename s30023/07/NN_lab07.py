import tensorflow as tf
import keras_tuner as kt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

def load_data(random_state=42):
    wine = fetch_ucirepo(id=109)
    print(wine.data.columns)
    X = wine.data.features
    y = wine.data.targets

    le = LabelEncoder()
    y = le.fit_transform(y.values.ravel())

    X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=random_state)

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_model(input_shape=(13,), output_units=3):
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=input_shape))

    model.add(tf.keras.layers.Dense(units=32, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.3))

    model.add(tf.keras.layers.Dense(units=16, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.2))

    model.add(tf.keras.layers.Dense(units=8, activation='relu', kernel_initializer='he_uniform'))
    model.add(tf.keras.layers.BatchNormalization())
    
    model.add(tf.keras.layers.Dense(output_units, activation='softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=["accuracy"],
    )

    return model

def evaluate_model(model, X_test, y_test):
    y_pred_prob = model.predict(X_test)
    y_pred = y_pred_prob.argmax(axis=1)

    print(
        f"""
        --- EVALUATION ---
        {classification_report(y_test, y_pred)}
        """)


def save_model(model, path='tensorflow_model.keras'):
    model.save(path)
    print(f'Model saved to {path}')


def main():
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()

    model = build_model()

    model.fit(X_train, y_train, epochs=60, batch_size=16, validation_data=(X_val, y_val))

    evaluate_model(model, X_test, y_test)

    save_model(model)

if __name__ == '__main__':
    main()