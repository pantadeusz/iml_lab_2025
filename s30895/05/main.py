from keras import Input, optimizers
from keras.src.saving.saving_lib import load_model
from keras_tuner import Hyperband
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, f1_score, precision_score, accuracy_score

import matplotlib.pyplot as plt
from keras import layers, models
from sklearn.model_selection import train_test_split
from pandas import DataFrame, read_csv


def loadData():
  return read_csv("creditcard_2023.csv")

def process_data(df):
  X = df.drop(['Class'],axis=1)
  y = df['Class']
  # print(X.head())
  X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=47)

  return X_train, X_test, y_train, y_test

def build_model_baseline(X_shape):
  model = models.Sequential([
    Input(shape=X_shape),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation="sigmoid")
  ])
  model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', 'precision', 'recall']
  )
  return model

def build_model_for_tuning(hp):
    model = models.Sequential()

    model.add(layers.Input(shape=(30,)))

    model.add(
      layers.Dense(
        units=hp.Int("units1", min_value=32, max_value=512, step=32),
        activation='relu'
      )
    )

    if hp.Boolean("dropout1"):
      model.add(layers.Dropout(rate=0.25))

    model.add(
      layers.Dense(
        units=hp.Int("units2", min_value=16, max_value=256, step=32),
        activation='relu'
      )
    )

    if hp.Boolean("dropout2"):
      model.add(layers.Dropout(rate=0.3))

    model.add(layers.Dense(1, activation="sigmoid"))

    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

    model.compile(
      optimizer=optimizers.Adam(learning_rate=learning_rate),
      loss="binary_crossentropy",
      metrics=["accuracy"],
    )

    return model



def evaluate_model(model, model_name, X_test, y_test):
  predictions = (model.predict(X_test) > 0.5).astype(int)

  print(f"Model: {model_name}")
  print("Accuracy:", accuracy_score(y_test, predictions))
  print("Precision:", precision_score(y_test, predictions))
  print("F1:", f1_score(y_test, predictions))

  cm = confusion_matrix(y_test, predictions)
  ConfusionMatrixDisplay(cm).plot()
  plt.title(model_name)
  plt.show()

def fit_and_save(model, file_name, X_train, y_train, batch, epochs):
  model.fit(X_train, y_train, batch_size = batch, epochs = epochs)
  model.save(file_name)
  print(f"model saved to {file_name}")




if __name__ == "__main__":

  X_train, X_test, y_train, y_test = process_data(loadData())

  logistic_regression = LogisticRegression()
  logistic_regression.fit(X_train,y_train)

  # fit_and_save(build_model_baseline((X_train.shape[1],)), "baseline_model.keras", X_train, y_train, 128, 12)
  baseline_model = load_model("baseline_model.keras")

  evaluate_model(logistic_regression, "logistic_regression", X_test, y_test)
  evaluate_model(baseline_model,"baseline_model",X_test, y_test)

  tuner = Hyperband(
    build_model_for_tuning,
    objective="val_accuracy",
    max_epochs=15,
    factor=3,
    directory="tuner_results",
    project_name="creditcard_fraud"
  )

  tuner.search(X_train, y_train,
               validation_split=0.2,
               epochs=15,
               batch_size=128)


  best_model = load_model('tuned_baseline_model.keras')
  evaluate_model(best_model, "tuned_baseline_model",X_test, y_test)
  # best_model = tuner.get_best_models(num_models=1)[0]
  # print(best_model)
  # best_model.save("tuned_baseline_model.keras")

  print(baseline_model.summary())
  print(best_model.summary())
