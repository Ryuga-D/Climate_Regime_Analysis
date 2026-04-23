import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model # type:ignore
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Concatenate # type:ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type:ignore
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import os
import random

# Reproducibility
def set_seeds(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.config.experimental.enable_op_determinism()

set_seeds(44)

# Sequence preparation

def prepare_sequences(data, sequence_length, forecast_horizon):
    X, y = [], []
    step_size = max(1, sequence_length // 3)
    for i in range(0, len(data) - sequence_length - forecast_horizon + 1, step_size):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length:i + sequence_length + forecast_horizon])
    print(f"Created {len(X)} overlapping sequences (step size: {step_size})")
    return np.array(X), np.array(y)

# Validation

def improved_validation(forecast_df, test_path):
    test_df = pd.read_csv(test_path)
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df.set_index('Date', inplace=True)
    forecast_df = forecast_df[test_df.columns]

    results = {}
    print("\nForecast Evaluation Metrics (vs Main_Test.csv):")
    for col in forecast_df.columns:
        true = test_df[col].values
        pred = forecast_df[col].values
        mae = mean_absolute_error(true, pred)
        r2 = r2_score(true, pred)
        true_trend = np.sign(np.diff(true))
        pred_trend = np.sign(np.diff(pred))
        trend_match = np.sum(true_trend == pred_trend)
        trend_acc = round((trend_match / len(true_trend)) * 100, 2)
        results[col] = {'MAE': round(mae, 4), 'R²': round(r2, 4), 'Trend_Accuracy_%': trend_acc}
        print(f"{col}: MAE = {mae:.4f}, R² = {r2:.4f}, Trend Accuracy = {trend_acc:.2f}%")
    return results

# Visualization

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

# Trend Analysis

def analyze_forecast_trends(forecast_df):
    trends = forecast_df.diff().mean()
    print("\n Forecast Trends (Mean Monthly Change):")
    print(trends.round(4))
    return trends

# Build Seq2Seq with Attention

def build_seq2seq_model(seq_len, horizon, features):
    encoder_inputs = Input(shape=(seq_len, features))
    encoder_lstm = LSTM(128, return_sequences=True, return_state=True)
    encoder_out, state_h, state_c = encoder_lstm(encoder_inputs)

    decoder_inputs = Input(shape=(horizon, features))
    decoder_lstm = LSTM(128, return_sequences=True, return_state=True)
    decoder_out, _, _ = decoder_lstm(decoder_inputs, initial_state=[state_h, state_c])

    attention = Attention()([decoder_out, encoder_out])
    concat = Concatenate()([decoder_out, attention])

    dense = Dense(features)(concat)

    return Model([encoder_inputs, decoder_inputs], dense)

# Main pipeline

def run_seq2seq_pipeline():
    print(" CLIMATE FORECASTING PIPELINE WITH SEQ2SEQ + ATTENTION")
    print("="*80)
    set_seeds(44)

    df = pd.read_csv('Main_df.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    features = ['Temp_Anomaly_°C', 'CO2_in_ppm', 'Precip_in_mm', 'Nino34_in_°C', 'Volcanic_Global', 'TSI_Wm2']
    data = df[features].values
    num_features = len(features)

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)

    sequence_length = 24
    forecast_horizon = 36

    train_data = data_scaled[df['Date'] <= '2013-12-31']
    X, y = prepare_sequences(train_data, sequence_length, forecast_horizon)

    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    model = build_seq2seq_model(sequence_length, forecast_horizon, num_features)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    model.summary()

    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

    history = model.fit([X_train, y_train], y_train, validation_data=([X_val, y_val], y_val),
                        epochs=100, batch_size=32, callbacks=[early_stop, reduce_lr], verbose=1)

    # Forecasting
    last_seq = train_data[-sequence_length:].reshape(1, sequence_length, num_features)
    decoder_start = np.zeros((1, forecast_horizon, num_features))
    prediction_scaled = model.predict([last_seq, decoder_start], verbose=0).reshape(forecast_horizon, num_features)
    prediction = scaler.inverse_transform(prediction_scaled)

    forecast_dates = pd.date_range(start='2014-01-01', end='2016-12-01', freq='MS')
    forecast_df = pd.DataFrame(prediction, columns=features, index=forecast_dates)

    print("\nForecast Summary:")
    print(forecast_df.describe().round(4))

    metrics = improved_validation(forecast_df, 'Main_Test.csv')
    visualize_forecasts(forecast_df, df)
    trends = analyze_forecast_trends(forecast_df)

    print("\n SEQ2SEQ + ATTENTION FORECASTING COMPLETED!")
    print("="*80)
    return forecast_df, model, trends, metrics

forecast_df, model, trends, metrics = run_seq2seq_pipeline()
print(metrics)