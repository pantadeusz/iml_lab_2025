import keras_tuner as kt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


data = load_breast_cancer()
X, y =  data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
input_shape = X_train_scaled.shape[1]

def build_model(hp):
    model = Sequential()

    hp_units_1 = hp.Int('units_1', min_value = 16, max_value=64, step=8)
    model.add(Dense(units=hp_units_1, activation='relu', input_shape=(input_shape,)))

    hp_dropout = hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)
    model.add(Dropout(rate=hp_dropout))

    hp_units_2 = hp.Int('units_2', min_value =8, max_value=32, step=8)
    model.add(Dense(units=hp_units_2, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    hp_learning_rate = hp.Choice('learning_rate', values=[1e-4, 1e-3, 1e-4])

    model.compile(
        optimizer=Adam(learning_rate=hp_learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model

tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='tune_results',
    project_name='breast_tune_results',
)

tuner.search(X_train_scaled, y_train, epochs=20, validation_split=0.2, verbose=0)

tuner.results_summary()

best_model = tuner.get_best_models(num_models=1)[0]

y_pred_proba = best_model.predict(X_test_scaled)
y_pred_dnn = (y_pred_proba >= 0.5).astype(int)

print("\n=== Deep Neural Network (DNN) po tuningu ===")
print(classification_report(y_test, y_pred_dnn, output_dict=True))

model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X_train_scaled, y_train)
y_pred_rf = model_rf.predict(X_test_scaled)

print("\n=== Random Forest Classifier (dla porównania) ===")
print(classification_report(y_test, y_pred_rf, output_dict=True))

