from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


data = load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# print(classification_report(y_test, y_pred, output_dict=True))

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
input_shape = X_train_scaled.shape[1]

model_dnn = Sequential([
    Dense(30, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),
    Dense(15, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_dnn.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model_dnn.fit(
    X_train_scaled, y_train,
    validation_split=0.1,
    epochs=50,
    batch_size=16,
    verbose=0
)

y_pred_proba = model_dnn.predict(X_test_scaled)
y_pred_dnn = (y_pred_proba >= 0.5).astype(int)

print("\n=== Deep Neural Network (DNN) ===")
print(classification_report(y_test, y_pred_dnn, output_dict=True))
print("\n=== Random Forest Classifier ===")
print(classification_report(y_test, y_pred, output_dict=True))
