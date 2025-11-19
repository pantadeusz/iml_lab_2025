import os
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from tensorflow.keras import layers, models, regularizers
# fetch dataset

wine = fetch_ucirepo(id=109)

# data (as pandas dataframes)

X = wine.data.features.values
y = wine.data.targets['class'].values.flatten() - 1

X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42, stratify=y
)

print("X_Train: ", X_train.shape, "y_train: ", y_train.shape)

rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_preds)

joblib.dump(rf_model, 'rfmodel.joblib')
rf_size_kb = os.path.getsize('rfmodel.joblib') / 1024


print(f"Random Forest Dokładność: {rf_acc * 100:.2f}%")
print(f"Rozmiar pliku modelu: {rf_size_kb:.2f} KB")
print("-" * 50)

model_simple = models.Sequential([
layers.Input(shape=(13,)),
layers.Dense(32, activation='relu'),
layers.Dense(16, activation='relu'),
layers.Dense(3, activation='softmax')
])

model_simple.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history_simple = model_simple.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test))
loss_simple, acc_simple = model_simple.evaluate(X_test, y_test, verbose=0)

print(f"NN (bez normalizacji) Dokładność: {acc_simple * 100:.2f}%")
print("-" *50)

normalizer = layers.Normalization(axis=-1)
normalizer.adapt(X_train)

model_norm = models.Sequential([
    layers.Input(shape=(13,)),
    normalizer,
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(3, activation='softmax')
])

model_norm.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model_norm.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test))
loss_norm, acc_norm = model_norm.evaluate(X_test, y_test, verbose=0)

print(f"NN (z normalizacją) Dokładność: {acc_norm*100:.2f}%")
print("-" *50)

configs = [
    {"layers": [64, 64, 64], "reg": None, "name": "Overkill_Deep"},
    {"layers": [128], "reg": None, "name": "Overkill_Wide"},
    {"layers": [32, 16, 8], "reg": None, "name": "Funnel_Classic"},
    {"layers": [4, 4, 4, 4], "reg": None, "name": "Deep_Narrow"},
    {"layers": [16], "reg": 0.1, "name": "Heavy_Reg"},
    {"layers": [7], "reg": None, "name": "Lucky_Seven"},
    {"layers": [16, 8], "reg": 0.001,"name": "Light_Reg"},
    {"layers": [],"reg": None, "name": "Linear_NoHidden"},
]
results = []

print(f"{'Nazwa':<15} | {'Acc %':<8} | {'Loss':<8} |{'Params':<8} | {'KB':<8}")
print("-" * 50)

for conf in configs:
    reg = regularizers.l2(conf["reg"]) if conf["reg"] else None

    model = models.Sequential()
    model.add(layers.Input(shape=(13,)))
    model.add(normalizer)

    if conf['layers']:
        for units in conf["layers"]:
            model.add(layers.Dense(units, activation='relu', kernel_regularizer=reg))

    model.add(layers.Dense(3, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.fit(X_train, y_train, epochs=100, verbose=0, validation_data=(X_test, y_test))
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    params = model.count_params()

    fname = f"model_{conf['name']}.keras"
    model.save(fname)
    size_kb = os.path.getsize(fname) / 1024
    print(f"{conf['name']:<15} | {acc * 100:05.2f}% | {loss:.4f} | {params:<8} | {size_kb:.2f}")
    results.append((conf["name"], acc, loss, params, size_kb))

print("-"*50)
best_model = max(results, key=lambda x: x[1])
smallest_perfect = min([r for r in results if r[1] >= 0.97], key=lambda x: x[3], default=None)

print(f"Model Random Forest: {rf_acc * 100:.2f}% ({rf_size_kb:.2f} KB)")
print(f"Najdokładniejszy NN: {best_model[0]} ({best_model[1] * 100:.2f}%)")
if smallest_perfect:
    print(f"Najmniejszy 'perfekcyjny' (>97%) NN: {smallest_perfect[0]} (tylko {smallest_perfect[3]} parametrów!)")
else:
    print("Żaden zminiaturyzowany model nie osiągnął >97%.")