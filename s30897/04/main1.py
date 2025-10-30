from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

data = load_diabetes()
X = data.data
y = data.target
n_features = X.shape[1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

scalar = StandardScaler()
X_train_scaled = scalar.fit_transform(X_train)
X_test_scaled = scalar.transform(X_test)

print("-"*30)
print(f"Liczba danych w zbiorze treningowym: {X_train_scaled.shape}")
print(f"Liczba danych w zbiorze testowym: {X_test_scaled.shape}")
print("-"*30)

# Random Forest

rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_scaled, y_train)
y_pred = rf_reg.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("-"*30)
print(f"Random forest - MSE: {mse:.2f} - R2: {r2:.4f}")
print("-"*30)

# DNN

DNN_Model = Sequential()
DNN_Model.add(Input(shape=(n_features,)))
DNN_Model.add(Dense(units=64, activation='relu'))
DNN_Model.add(Dense(units=32, activation='relu'))
DNN_Model.add(Dense(units=1))

DNN_Model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])

history = DNN_Model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)

loss_dnn, mae_dnn = DNN_Model.evaluate(X_test_scaled, y_test, verbose=0)
y_pred_dnn = DNN_Model.predict(X_test_scaled)
r2_dnn = r2_score(y_test, y_pred_dnn)

print("-"*30)

print(f"Model DNN - MSE: {loss_dnn:.2f} - R2: {r2_dnn:.4f}")

print("-"*30)

if r2 > r2_dnn:
    print("Wniosek: Random Forest poradził sobie lepiej na tym zbiorze danych.")
elif r2_dnn > r2:
    print("Wniosek: Sieć neuronowa (DNN) poradziła sobie lepiej.")
else:
    print("Wniosek: Oba modele osiągnęły bardzo podobny wynik.")

print("-"*30)