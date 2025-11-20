import keras
from keras import regularizers
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from keras import layers
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,f1_score
from keras import Model, models

wine = fetch_ucirepo(id=109)

X = wine.data.features
y = wine.data.targets.squeeze()

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,train_size=0.7,random_state=42)

y_train = y_train - 1
y_test = y_test - 1

def get_predictions(classifier,X_test):
    predictions = classifier.predict(X_test)
    return predictions

def get_metrics(predictions, y_test):
    cm = confusion_matrix(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average="macro")
    f1 = f1_score(y_test,predictions,average="macro")
    return cm,accuracy,precision,f1

def plot_accuracy_and_validation(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy.png")
    plt.show()


    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss', linestyle='--')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("loss.png")
    plt.show()

def build_basic_dnn(X_train_shape):
    model = models.Sequential([
        keras.Input(shape=(X_train_shape,)),
        layers.Dense(32, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),

        layers.Dense(16, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),

        layers.Dense(3, activation='softmax')
    ])


    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def print_metrics(f1,accuracy,precision,cm,classifier_name):
    print(f"__{classifier_name} performance__")
    print("confusion matrix: \n", cm)
    print("accuracy: ", accuracy)
    print("precision: ", precision)
    print("f1 score: ", f1)

def fit_and_save(X_train,y_train,X_test,y_test):
    # dnn
    basic_dnn = build_basic_dnn(X_train.shape[1])
    history = basic_dnn.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=200,
        batch_size=8,
        shuffle=True,
    )

    # basic_dnn.save("models/baseline_with_l2.keras")
    return basic_dnn, history

# Random Forest
random_forest = RandomForestClassifier()
random_forest.fit(X_train,y_train)
random_forest_predictions = get_predictions(random_forest,X_test)
forest_cm, forest_accuracy, forest_precision, forest_f1 = get_metrics(random_forest_predictions,y_test)

# dnn
basic_dnn, history = fit_and_save(X_train,y_train,X_test,y_test)
dnn_predictions = get_predictions(basic_dnn,X_test).argmax(axis=1)
dnn_cm, dnn_accuracy, dnn_precision, dnn_f1 = get_metrics(dnn_predictions,y_test)

# metrics
basic_dnn.summary()
print_metrics(forest_f1, forest_accuracy, forest_precision, forest_cm, "Random Forest")
print_metrics(dnn_f1, dnn_accuracy, dnn_precision, dnn_cm, "dnn")
plot_accuracy_and_validation(history)