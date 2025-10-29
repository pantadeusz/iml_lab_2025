from keras import Sequential
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import Input
from tensorflow.python.keras.layers import Dense
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import matplotlib.pyplot as plt


import tensorflow as tf
print("TensorFlow version:", tf.__version__)
mushroom = fetch_ucirepo(id=2)

def prepare_data(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_encoded = encoder.fit_transform(X_train)
    X_test_encoded = encoder.transform(X_test)
    # print(X_train.head())
    print(X_train_encoded)

    le_target = LabelEncoder()
    y_train_encoded = le_target.fit_transform(y_train.values.ravel())
    y_test_encoded = le_target.transform(y_test.values.ravel())

    print(X_train_encoded)

    return X_train_encoded,X_test_encoded, y_train_encoded, y_test_encoded

def predict_with_Random_Forest():
    model_random_forest = RandomForestClassifier(n_estimators=100, random_state=42, )
    model_random_forest.fit(X_train_encoded, y_train_encoded)
    return model_random_forest.predict(X_test_encoded)

def show_metrics(y_pred, y_true):
    f1 = f1_score(y_test_encoded, y_pred)
    print(f"F1 score: {f1:.4f}")
    cm = confusion_matrix(y_test_encoded, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.savefig("confusion_matrix.png")
    plt.close()

X_train_encoded,X_test_encoded, y_train_encoded, y_test_encoded = prepare_data(mushroom.data.features,mushroom.data.targets)

y_pred = predict_with_Random_Forest()

show_metrics(y_pred,y_test_encoded)

# X_train_encoded_reshaped = X_train_encoded[:1].reshape((1, 1, 117))


model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(Input(X_train_encoded)),
    tf.keras.layers.Dense(128,activation='relu'),
    tf.keras.layers.Dense(10)
])

# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10)
# ])
predictions = model(X_train_encoded[:1]).numpy()

