import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

classic_model = RandomForestClassifier(random_state=42)

classic_model.fit(X_train, y_train)
y_pred_classic = classic_model.predict(X_test)

classic_report = classification_report(y_test, y_pred_classic, output_dict=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

nn_model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(30,)),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(32, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    # tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(
    optimizer="adam",
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
    ]
)

training = nn_model.fit(X_train, y_train, epochs=30, batch_size=16, validation_split=0.1, verbose=0)
history_df = pd.DataFrame(training.history)

history_df.index = history_df.index + 1
history_df.index.name = 'Epoka'

history_df.to_csv("training.csv", index=False)

y_pred_nn = (nn_model.predict(X_test) > 0.5).astype("int32")

nn_report = classification_report(y_test, y_pred_nn, output_dict=True)

print(classic_report)
print(nn_report)