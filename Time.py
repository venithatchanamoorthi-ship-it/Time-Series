# ============================================================
# Advanced Time Series Forecasting with Deep Learning (LSTM)
# Complete, End-to-End, Runnable Python Program
# ============================================================
# - Synthetic dataset generation (>=500 observations)
# - Preprocessing: scaling, windowing
# - Baseline ARIMA model
# - Deep Learning Seq2Seq LSTM model (TensorFlow/Keras)
# - Hyperparameter tuning (Grid Search)
# - Early stopping & model checkpointing
# - Evaluation with MAE, RMSE, MAPE, Directional Accuracy
# - Multi-step forecasting visualization
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import itertools
import os

# -----------------------------
# 1. Reproducibility
# -----------------------------
np.random.seed(42)
tf.random.set_seed(42)

# -----------------------------
# 2. Dataset Generation (Synthetic Time Series)
# -----------------------------
N = 1000  # >= 500 observations
time = np.arange(N)
trend = time * 0.01
seasonality = 5 * np.sin(2 * np.pi * time / 50)
noise = np.random.normal(scale=1.0, size=N)
series = trend + seasonality + noise

df = pd.DataFrame({"value": series})

# -----------------------------
# 3. Train / Validation / Test Split
# -----------------------------
train_size = int(0.7 * N)
val_size = int(0.15 * N)

t_train = df.iloc[:train_size]
t_val = df.iloc[train_size:train_size + val_size]
t_test = df.iloc[train_size + val_size:]

# -----------------------------
# 4. Scaling
# -----------------------------
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(t_train)
val_scaled = scaler.transform(t_val)
test_scaled = scaler.transform(t_test)

# -----------------------------
# 5. Windowing Function
# -----------------------------
def create_sequences(data, look_back=30, horizon=5):
    X, y = [], []
    for i in range(len(data) - look_back - horizon + 1):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + horizon])
    return np.array(X), np.array(y)

# -----------------------------
# 6. Baseline Model – ARIMA
# -----------------------------
arima_model = ARIMA(t_train["value"], order=(5, 1, 0))
arima_fit = arima_model.fit()
arima_forecast = arima_fit.forecast(steps=len(t_test))

# -----------------------------
# 7. Evaluation Metrics
# -----------------------------
def mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def directional_accuracy(y_true, y_pred):
    return np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))

# -----------------------------
# 8. Hyperparameter Grid
# -----------------------------
param_grid = {
    "look_back": [20, 30],
    "units": [32, 64],
    "dropout": [0.2],
    "batch_size": [32],
    "epochs": [30]
}

best_score = np.inf
best_params = None
best_model = None

# -----------------------------
# 9. Grid Search
# -----------------------------
for params in itertools.product(*param_grid.values()):
    look_back, units, dropout, batch_size, epochs = params

    X_train, y_train = create_sequences(train_scaled, look_back)
    X_val, y_val = create_sequences(val_scaled, look_back)

    model = Sequential([
        LSTM(units, input_shape=(look_back, 1)),
        Dropout(dropout),
        Dense(y_train.shape[1])
    ])

    model.compile(optimizer="adam", loss="mse")

    es = EarlyStopping(patience=5, restore_best_weights=True)

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[es],
        verbose=0
    )

    val_pred = model.predict(X_val)
    val_pred_inv = scaler.inverse_transform(val_pred.reshape(-1, 1)).reshape(val_pred.shape)
    y_val_inv = scaler.inverse_transform(y_val.reshape(-1, 1)).reshape(y_val.shape)

    rmse = np.sqrt(mean_squared_error(y_val_inv.flatten(), val_pred_inv.flatten()))

    if rmse < best_score:
        best_score = rmse
        best_params = params
        best_model = model

# -----------------------------
# 10. Final Training with Best Params
# -----------------------------
look_back, units, dropout, batch_size, epochs = best_params

X_train_full, y_train_full = create_sequences(
    np.vstack([train_scaled, val_scaled]), look_back
)

checkpoint_path = "best_lstm_model.h5"

checkpoint = ModelCheckpoint(
    checkpoint_path, monitor="loss", save_best_only=True
)

final_model = Sequential([
    LSTM(units, input_shape=(look_back, 1)),
    Dropout(dropout),
    Dense(y_train_full.shape[1])
])

final_model.compile(optimizer="adam", loss="mse")

final_model.fit(
    X_train_full, y_train_full,
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint],
    verbose=1
)

# -----------------------------
# 11. Test Evaluation
# -----------------------------
X_test, y_test = create_sequences(test_scaled, look_back)

test_pred = final_model.predict(X_test)

test_pred_inv = scaler.inverse_transform(test_pred.reshape(-1, 1)).reshape(test_pred.shape)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).reshape(y_test.shape)

mae = mean_absolute_error(y_test_inv.flatten(), test_pred_inv.flatten())
rmse = np.sqrt(mean_squared_error(y_test_inv.flatten(), test_pred_inv.flatten()))
mape_val = mape(y_test_inv.flatten(), test_pred_inv.flatten())
da = directional_accuracy(y_test_inv.flatten(), test_pred_inv.flatten())

print("Final Optimized Hyperparameters:")
print(f"Look-back={look_back}, Units={units}, Dropout={dropout}")
print("\nTest Metrics:")
print(f"MAE  : {mae:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"MAPE : {mape_val:.2f}%")
print(f"Directional Accuracy: {da:.2f}")

# -----------------------------
# 12. Visualization
# -----------------------------
plt.figure()
plt.plot(y_test_inv.flatten(), label="Actual")
plt.plot(test_pred_inv.flatten(), label="LSTM Forecast")
plt.title("Multi-step Forecast vs Actual")
plt.legend()
plt.show()

plt.figure()
plt.plot(t_test.index[:len(arima_forecast)], t_test["value"].values[:len(arima_forecast)], label="Actual")
plt.plot(t_test.index[:len(arima_forecast)], arima_forecast.values, label="ARIMA Forecast")
plt.title("Baseline ARIMA vs Actual")
plt.legend()
plt.show()
