import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# 1. DATA GENERATION (Synthetic for immediate execution)
def generate_time_series(n_points=2000):
    time = np.arange(n_points)
    # Trend + Seasonality + Noise
    series = 0.5 * time + 50 * np.sin(time / 10) + np.random.normal(0, 10, n_points)
    return series

data = generate_time_series()
df = pd.DataFrame(data, columns=['value'])

# 2. FEATURE ENGINEERING & PREPROCESSING
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df)

def create_windowed_dataset(dataset, look_back=60):
    X, y = [], []
    for i in range(len(dataset) - look_back):
        X.append(dataset[i:(i + look_back), 0])
        y.append(dataset[i + look_back, 0])
    return np.array(X), np.array(y)

LOOK_BACK = 60
X, y = create_windowed_dataset(scaled_data, LOOK_BACK)

# Reshape for LSTM: [samples, time steps, features]
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split into Train/Test
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 3. MODEL ARCHITECTURE (Optimized LSTM)
model = Sequential([
    Input(shape=(LOOK_BACK, 1)),
    LSTM(units=64, return_sequences=True),
    Dropout(0.2),
    LSTM(units=64, return_sequences=False),
    Dropout(0.2),
    Dense(units=32, activation='relu'),
    Dense(units=1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 4. TRAINING WITH CALLBACKS (Model Checkpointing)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint('best_model.keras', save_best_only=True)
]

print("Starting training...")
history = model.fit(
    X_train, y_train, 
    epochs=20, 
    batch_size=32, 
    validation_split=0.1, 
    callbacks=callbacks,
    verbose=1
)

# 5. EVALUATION & FORECAST BIAS ANALYSIS
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_unscaled = scaler.inverse_transform(y_test.reshape(-1, 1))

# Metrics
rmse = np.sqrt(mean_squared_error(y_test_unscaled, predictions))
mape = mean_absolute_percentage_error(y_test_unscaled, predictions)
bias = np.mean(predictions - y_test_unscaled) # Positive = overforecasting

print(f"\n--- Model Performance ---")
print(f"RMSE: {rmse:.4f}")
print(f"MAPE: {mape:.4f}")
print(f"Forecast Bias: {bias:.4f}")

# 6. VISUALIZATION
plt.figure(figsize=(12, 6))
plt.plot(y_test_unscaled, label='Actual Values', color='blue')
plt.plot(predictions, label='LSTM Predictions', color='red', linestyle='--')
plt.title('Time Series Forecasting Results')
plt.legend()
plt.show()
