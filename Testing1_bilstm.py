import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model # type:ignore
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Reshape # type:ignore
from tensorflow.keras.optimizers import Adam # type:ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type:ignore
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import os
import random

def set_seeds(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()

set_seeds(44)

def prepare_sequences(data, sequence_length, forecast_horizon):
    X, y = [], []
    step_size = max(1, sequence_length // 3)
    for i in range(0, len(data) - sequence_length - forecast_horizon + 1, step_size):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + forecast_horizon])
    print(f"Created {len(X)} overlapping sequences (step size: {step_size})")
    return np.array(X), np.array(y)

def build_lstm_model(input_shape, forecast_horizon, num_features):
    from tensorflow.keras.layers import Bidirectional # type:ignore

    inputs = Input(shape=input_shape)

    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, activation='relu', return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    x = Bidirectional(LSTM(64, activation='relu', return_sequences=False))(x)
    x = Dropout(0.3)(x)

    # Dense + Reshape output
    output_dim = forecast_horizon * num_features
    outputs = Dense(output_dim, activation='linear')(x)
    outputs = Reshape((forecast_horizon, num_features))(outputs)

    return Model(inputs=inputs, outputs=outputs)


def improved_validation(forecast_df, test_path):
    test_df = pd.read_csv(test_path)
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df.set_index('Date', inplace=True)

    # Align forecast columns with test columns
    forecast_df = forecast_df[test_df.columns]

    results = {}

    print("\n📊 Forecast Evaluation Metrics (vs Main_Test.csv):")
    for col in forecast_df.columns:
        true = test_df[col].values
        pred = forecast_df[col].values

        # Standard metrics
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)

        # Trend accuracy: % of times the direction matches (up/down)
        true_trend = np.sign(np.diff(true))
        pred_trend = np.sign(np.diff(pred))
        trend_match = np.sum(true_trend == pred_trend)
        trend_acc = round((trend_match / len(true_trend)) * 100, 2)

        results[col] = {
            'MAE': round(mae, 4),
            'R²': round(r2, 4),
            'Trend_Accuracy_%': trend_acc
        }

        print(f"{col}: MAE = {mae:.4f}, R² = {r2:.4f}, Trend Accuracy = {trend_acc:.2f}%")

    return results


def visualize_forecasts(forecast_df, original_df):
    plt.figure(figsize=(15, 8))
    for col in forecast_df.columns:
        plt.plot(original_df['Date'], original_df[col], label=f"{col} (Train)")
        plt.plot(forecast_df.index, forecast_df[col], '--', label=f"{col} (Forecast)")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.title("Climate Forecast vs Historical")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def analyze_forecast_trends(forecast_df):
    trends = forecast_df.diff().mean()
    print("\n📈 Forecast Trends (Mean Monthly Change):")
    print(trends.round(4))
    return trends

def run_lstm_pipeline():
    print("🌍 CLIMATE FORECASTING PIPELINE WITH PURE LSTM")
    print("="*80)

    set_seeds(44)

    # Load and preprocess data
    c_df = pd.read_csv('Main_df.csv')
    c_df['Date'] = pd.to_datetime(c_df['Date'])
    
    features = ['Temp_Anomaly_°C', 'CO2_in_ppm', 'Precip_in_mm',
                'Nino34_in_°C', 'Volcanic_Global', 'TSI_Wm2']
    
    data = c_df[features].values
    num_features = len(features)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    sequence_length = 24
    forecast_horizon = 36

    train_end_date = pd.to_datetime('2013-12-31')
    train_data_indices = c_df[c_df['Date'] <= train_end_date].index
    train_data_scaled = data_scaled[train_data_indices]

    X, y = prepare_sequences(train_data_scaled, sequence_length, forecast_horizon)

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"📊 Train/Validation Split:")
    print(f"   Training X shape: {X_train.shape}, y shape: {y_train.shape}")
    print(f"   Validation X shape: {X_val.shape}, y shape: {y_val.shape}")

    model = build_lstm_model(
        input_shape=(sequence_length, num_features),
        forecast_horizon=forecast_horizon,
        num_features=num_features
    )
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    model.summary()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=7, min_lr=1e-7, verbose=1)
    ]
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Plot training history
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title("Loss over Epochs")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='MAE')
    plt.plot(history.history['val_mae'], label='Val MAE')
    plt.title("MAE over Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Forecast
    last_sequence = train_data_scaled[-sequence_length:].reshape(1, sequence_length, num_features)
    predicted_scaled = model.predict(last_sequence, verbose=0)
    predicted_scaled = predicted_scaled.reshape(forecast_horizon, num_features)
    forecast_climate = scaler.inverse_transform(predicted_scaled)

    forecast_dates = pd.date_range(start='2014-01-01', end='2016-12-01', freq='MS')
    forecast_df = pd.DataFrame(forecast_climate, columns=features, index=forecast_dates)

    print(f"\n📊 Forecast Summary:")
    print(forecast_df.describe().round(4))

    # Validation & visualization
    metrics = improved_validation(forecast_df, 'Main_Test.csv')
    visualize_forecasts(forecast_df, c_df)
    trends = analyze_forecast_trends(forecast_df)

    print("\n🎉 PURE LSTM FORECASTING COMPLETED!")
    print("="*80)
    return forecast_df, model, trends, metrics


forecast_df, model, trends, metrics = run_lstm_pipeline()
print("whudhi")
print(metrics)