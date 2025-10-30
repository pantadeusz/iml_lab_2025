from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

def train_models(test_size):
    data = load_diabetes()
    X = data.data
    y = data.target
    n_features = X.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Random Forest
    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_train_scaled, y_train)
    y_pred_rf = rf_reg.predict(X_test_scaled)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    # DNN
    DNN_Model = Sequential()
    DNN_Model.add(Input(shape=(n_features,)))
    DNN_Model.add(Dense(units=64, activation='relu'))
    DNN_Model.add(Dense(units=32, activation='relu'))
    DNN_Model.add(Dense(units=1))
    DNN_Model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
    DNN_Model.fit(X_train_scaled, y_train, epochs=100, batch_size=32, verbose=0)
    y_pred_dnn = DNN_Model.predict(X_test_scaled)
    mse_dnn = mean_squared_error(y_test, y_pred_dnn)
    r2_dnn = r2_score(y_test, y_pred_dnn)

    return {"test_size": test_size, "rf": {"mse": mse_rf, "r2": r2_rf}, "dnn": {"mse": mse_dnn, "r2": r2_dnn}}

# Testowanie różnych test_size
test_sizes = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99]
for ts in test_sizes:
    results = train_models(ts)
    print("-"*50)
    print(f"test_size = {results['test_size']}")
    print(f"Random Forest - MSE: {results['rf']['mse']:.2f}, R2: {results['rf']['r2']:.4f}")
    print(f"DNN           - MSE: {results['dnn']['mse']:.2f}, R2: {results['dnn']['r2']:.4f}")
