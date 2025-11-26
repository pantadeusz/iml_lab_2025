import kagglehub
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from sklearn.model_selection import train_test_split

# 1. Wczytanie danych
path = kagglehub.dataset_download("uciml/pima-indians-diabetes-database")
data = pd.read_csv(path + "/diabetes.csv")

print(data.shape)

# Podział na cechy i etykiety
X = data.drop('Outcome', axis=1).values
y = data['Outcome'].values

# W TensorFlow nie musisz konwertować na tensory ręcznie — model.fit zrobi to automatycznie.
# TensorFlow sam obsługuje batch’e i epoki w funkcji fit().

# 2. Podział na train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Definicja modelu
model = models.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(12, activation='relu'),  # warstwa ukryta 1
    layers.Dense(8, activation='relu'),  # warstwa ukryta 2
    layers.Dense(1, activation='sigmoid')  # wyjście: prawdopodobieństwo 0/1
])

# Dense to odpowiednik nn.Linear w PyTorch.
# Nie trzeba pisać klasy z forward — TensorFlow automatycznie buduje graf przepływu danych.

# 4. Kompilacja modelu
model.compile(
    optimizer=optimizers.Adam( learning_rate=0.001),
    loss=losses.BinaryCrossentropy(),
    metrics=['accuracy']
)

# TensorFlow tworzy wewnętrzny graf obliczeń i zarządza gradientami.
# Nie musisz ręcznie wywoływać loss.backward() ani optimizer.step().


# 6. Trenowanie modelu
history = model.fit(
    X_train, y_train,
    epochs=300,
    batch_size=10,
    verbose=1 # pokaż postęp
)

# Nie trzeba pisać dwóch pętli (for epoch + batch) — fit sam dzieli dane na batch’e i iteruje po epokach.
# TensorFlow obsługuje gradienty, aktualizacje wag i monitorowanie loss/accuracy automatycznie.

# 7. Ewaluacja modelu
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
