import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # type: ignore
from tensorflow.keras.regularizers import l1_l2 # type: ignore
from sklearn.preprocessing import StandardScaler
import random
import os

def set_seeds(seed=42):
    """Set seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For TensorFlow 2.x deterministic operations
    tf.config.experimental.enable_op_determinism()

# Set seeds for reproducibility
set_seeds(44)

def build_improved_autoencoder(input_dim, encoding_dim=3):
    """
    Improved autoencoder architecture to fix overfitting
    """
    # Input layer
    input_layer = Input(shape=(input_dim,))
    
    # Encoder with stronger regularization
    encoded = Dense(32, activation='relu',
                   kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(input_layer)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)  # Increased dropout
    
    encoded = Dense(16, activation='relu',
                   kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(encoded)
    encoded = BatchNormalization()(encoded)
    encoded = Dropout(0.3)(encoded)
    
    # Bottleneck (latent representation)
    encoded = Dense(encoding_dim, activation='linear', name='latent_space')(encoded)
    
    # Decoder with regularization
    decoded = Dense(16, activation='relu',
                   kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(encoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.3)(decoded)
    
    decoded = Dense(32, activation='relu',
                   kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(decoded)
    decoded = BatchNormalization()(decoded)
    decoded = Dropout(0.3)(decoded)
    
    decoded = Dense(input_dim, activation='linear')(decoded)
    
    autoencoder = Model(input_layer, decoded)
    encoder = Model(input_layer, encoded)

    # CREATE SEPARATE DECODER MODEL
    decoder_input = Input(shape=(encoding_dim,))
    
    # Decoder layers (same as above but starting from latent input)
    decoder_hidden = Dense(16, activation='relu',
                          kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(decoder_input)
    decoder_hidden = BatchNormalization()(decoder_hidden)
    decoder_hidden = Dropout(0.3)(decoder_hidden)
    
    decoder_hidden = Dense(32, activation='relu',
                          kernel_regularizer=l1_l2(l1=0.001, l2=0.001))(decoder_hidden)
    decoder_hidden = BatchNormalization()(decoder_hidden)
    decoder_hidden = Dropout(0.3)(decoder_hidden)
    
    decoder_output = Dense(input_dim, activation='linear')(decoder_hidden)
    
    decoder = Model(decoder_input, decoder_output)
    
    return autoencoder, encoder, decoder

def train_improved_autoencoder(df_scaled, encoding_dim=3):
    """
    Train improved autoencoder with better hyperparameters
    """
    print("=== TRAINING IMPROVED AUTOENCODER ===")
    
    # Build autoencoder
    autoencoder, encoder, decoder = build_improved_autoencoder(df_scaled.shape[1], encoding_dim)
    
    # Compile with lower learning rate
    autoencoder.compile(optimizer=Adam(learning_rate=0.0005), 
                       loss='mse', 
                       metrics=['mae'])
    
    print("Improved Autoencoder Architecture:")
    autoencoder.summary()
    
    # Enhanced callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7)
    ]
    
    # Train with more epochs and smaller batch size
    history = autoencoder.fit(df_scaled, df_scaled,
                             epochs=200,
                             batch_size=32,
                             validation_split=0.2,
                             callbacks=callbacks,
                             verbose=1)
    
    # 🎯 COPY WEIGHTS FROM AUTOENCODER TO SEPARATE DECODER
    print("\n🔧 Copying decoder weights from trained autoencoder...")
    
    # Find decoder layers in autoencoder (after latent layer)
    autoencoder_layers = autoencoder.layers
    latent_idx = -1
    
    # Find latent layer
    for i, layer in enumerate(autoencoder_layers):
        if hasattr(layer, 'name') and 'latent' in layer.name.lower():
            latent_idx = i
            break
    
    if latent_idx == -1:
        # Find bottleneck by smallest units
        min_units = float('inf')
        for i, layer in enumerate(autoencoder_layers):
            if hasattr(layer, 'units') and layer.units < min_units:
                min_units = layer.units
                latent_idx = i
    
    print(f"   Found latent layer at index: {latent_idx}")
    
    # Copy weights from autoencoder decoder to separate decoder
    try:
        decoder_layers_ae = autoencoder_layers[latent_idx + 1:]  # Decoder layers in autoencoder
        decoder_layers_sep = decoder.layers[1:]  # Skip input layer in separate decoder
        
        copied_layers = 0
        for ae_layer, sep_layer in zip(decoder_layers_ae, decoder_layers_sep):
            if hasattr(ae_layer, 'get_weights') and len(ae_layer.get_weights()) > 0:
                if hasattr(sep_layer, 'set_weights'):
                    try:
                        sep_layer.set_weights(ae_layer.get_weights())
                        copied_layers += 1
                        print(f"   ✅ Copied weights: {ae_layer.name} → {sep_layer.name}")
                    except Exception as e:
                        print(f"   ⚠️ Failed to copy {ae_layer.name}: {e}")
        
        print(f"   🎯 Successfully copied {copied_layers} decoder layers")
        
    except Exception as e:
        print(f"   ❌ Weight copying failed: {e}")

    # Enhanced plotting
    plt.figure(figsize=(15, 10))
    
    # Loss plot
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    plt.title('Improved Autoencoder - Training Loss', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # MAE plot
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE', linewidth=2)
    plt.plot(history.history['val_mae'], label='Validation MAE', linewidth=2)
    plt.title('Improved Autoencoder - Training MAE', fontsize=14)
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Performance metrics
    final_train_mae = history.history['mae'][-1]
    final_val_mae = history.history['val_mae'][-1]
    overfitting_gap = final_val_mae - final_train_mae
    
    plt.subplot(2, 2, 3)
    metrics = ['Train MAE', 'Val MAE', 'Overfitting Gap']
    values = [final_train_mae, final_val_mae, overfitting_gap]
    colors = ['green', 'orange', 'red' if overfitting_gap > 0.2 else 'yellow']
    
    bars = plt.bar(metrics, values, color=colors, alpha=0.7)
    plt.title('Final Performance Metrics', fontsize=14)
    plt.ylabel('MAE')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Reconstruction quality analysis
    plt.subplot(2, 2, 4)
    reconstructed = autoencoder.predict(df_scaled)
    reconstruction_errors = np.mean(np.square(df_scaled - reconstructed), axis=1)
    
    plt.hist(reconstruction_errors, bins=30, alpha=0.7, color='purple')
    plt.axvline(np.mean(reconstruction_errors), color='red', linestyle='--',
                label=f'Mean Error: {np.mean(reconstruction_errors):.3f}')
    plt.title('Reconstruction Error Distribution', fontsize=14)
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Performance summary
    print("\n" + "="*50)
    print("🎯 PERFORMANCE SUMMARY")
    print("="*50)
    print(f"📊 Final Training MAE: {final_train_mae:.4f}")
    print(f"📊 Final Validation MAE: {final_val_mae:.4f}")
    print(f"📊 Overfitting Gap: {overfitting_gap:.4f}")
    
    if overfitting_gap < 0.1:
        print("✅ EXCELLENT: Minimal overfitting!")
    elif overfitting_gap < 0.2:
        print("🟡 GOOD: Acceptable overfitting level")
    else:
        print("🔴 CONCERN: High overfitting detected")
    
    print(f"📊 Mean Reconstruction Error: {np.mean(reconstruction_errors):.4f}")
    print(f"📊 Std Reconstruction Error: {np.std(reconstruction_errors):.4f}")
    
    return autoencoder, encoder, decoder

def evaluate_autoencoder_quality(autoencoder, encoder, df_scaled, df, c_df):
    """
    Comprehensive evaluation of autoencoder quality
    """
    print("\n" + "="*50)
    print("🔍 AUTOENCODER QUALITY EVALUATION")
    print("="*50)
    
    # Get reconstructions and latent features
    reconstructed = autoencoder.predict(df_scaled)
    latent_features = encoder.predict(df_scaled)
    
    # 1. Reconstruction Quality per Variable
    reconstruction_quality = {}
    
    plt.figure(figsize=(15, 10))
    
    for i, col in enumerate(df.columns):
        original = df_scaled[:, i]
        recon = reconstructed[:, i]
        
        # Calculate metrics
        mse = np.mean(np.square(original - recon))
        mae = np.mean(np.abs(original - recon))
        correlation = np.corrcoef(original, recon)[0, 1]
        
        reconstruction_quality[col] = {
            'MSE': mse,
            'MAE': mae,
            'Correlation': correlation
        }
        
        # Plot original vs reconstructed
        plt.subplot(2, 3, i+1)
        plt.scatter(original, recon, alpha=0.6, s=20)
        plt.plot([original.min(), original.max()], [original.min(), original.max()], 
                'r--', linewidth=2)
        plt.xlabel('Original')
        plt.ylabel('Reconstructed')
        plt.title(f'{col}\nCorr: {correlation:.3f}, MAE: {mae:.3f}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 2. Print reconstruction quality summary
    print("\n📊 RECONSTRUCTION QUALITY BY VARIABLE:")
    print("-" * 60)
    for var, metrics in reconstruction_quality.items():
        print(f"{var:20s} | MAE: {metrics['MAE']:.4f} | Corr: {metrics['Correlation']:.4f}")
    
    # 3. Latent space analysis
    print(f"\n🧠 LATENT SPACE ANALYSIS:")
    print(f"Latent dimensions: {latent_features.shape[1]}")
    print(f"Latent variance per dimension:")
    for i in range(latent_features.shape[1]):
        var = np.var(latent_features[:, i])
        print(f"  Latent {i+1}: {var:.4f}")

    return;
def extract_latent_features(encoder, df_scaled, c_df,df):
    """
    Extract latent representations and analyze them
    """
    print("=== EXTRACTING LATENT FEATURES ===")
    
    # Get latent representations
    latent_features = encoder.predict(df_scaled)
    
    # Create DataFrame with latent features
    latent_df = pd.DataFrame(latent_features, 
                           columns=[f'Latent_{i+1}' for i in range(latent_features.shape[1])])
    latent_df['Date'] = c_df['Date'].values
    latent_df['Date'] = pd.to_datetime(latent_df['Date'])
    print(f"Latent space shape: {latent_features.shape}")
    
    # Correlation analysis
    original_df = pd.DataFrame(df_scaled, columns=df.columns)
    combined_df = pd.concat([original_df, latent_df.drop(['Date'], axis=1)], axis=1)
    correlation_matrix = combined_df.corr()
    
    # Plot correlation between original and latent features
    plt.figure(figsize=(12, 8))
    latent_corr = correlation_matrix.loc[df.columns, [f'Latent_{i+1}' for i in range(latent_features.shape[1])]]
    sns.heatmap(latent_corr, annot=True, cmap='RdBu_r', center=0, fmt='.3f')
    plt.title('Correlation: Original Variables vs Latent Features')
    plt.tight_layout()
    plt.show()
    
    # Time series plots of latent features
    plt.figure(figsize=(15, 10))
    for i in range(latent_features.shape[1]):
        plt.subplot(latent_features.shape[1], 1, i+1)
        plt.plot(latent_df['Date'], latent_df[f'Latent_{i+1}'], 'b-', alpha=0.7)
        plt.title(f'Latent Feature {i+1} Over Time')
        plt.xlabel('Date')
        plt.ylabel(f'Latent_{i+1}')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return latent_df, latent_features
def main():
    # Load your data (same as before)
    c_df = pd.read_csv('Main_df.csv')
    c_df['Date'] = pd.to_datetime(c_df['Date'])
    c_df['Decade'] = (c_df['Date'].dt.year // 10) * 10
    
    df = c_df[['Temp_Anomaly_°C', 'CO2_in_ppm', 'Precip_in_mm', 'Nino34_in_°C', 'Volcanic_Global', 'TSI_Wm2']]
    
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df)
    
    # Train improved autoencoder
    autoencoder, encoder, decoder = train_improved_autoencoder(df_scaled, encoding_dim=3)
    
    # Evaluate quality
    evaluate_autoencoder_quality(autoencoder, encoder, df_scaled, df, c_df)
    
    # Extract latent features (your existing function)
    latent_df, latent_features = extract_latent_features(encoder, df_scaled, c_df, df)
    
    return autoencoder, encoder, decoder, latent_df, latent_features

# Function to set different seeds for reproducibility and compare the results
# def run_multiple_experiments(num_runs=5):
#     """Run multiple training experiments with different seeds"""
#     results = []
    
#     for i in range(num_runs):
#         print(f"\n{'='*60}")
#         print(f"🔄 EXPERIMENT RUN {i+1}/{num_runs} (Seed: {42 + i})")
#         print(f"{'='*60}")
        
#         # Run main with different seed
#         set_seeds(42 + i)
#         autoencoder, encoder, latent_df, latent_features = main()
        
#         # Calculate metrics
#         c_df = pd.read_csv('Main_df.csv')
#         c_df['Date'] = pd.to_datetime(c_df['Date'])
#         df = c_df[['Temp_Anomaly_°C', 'CO2_in_ppm', 'Precip_in_mm', 'Nino34_in_°C', 'Volcanic_Global', 'TSI_Wm2']]
        
#         scaler = StandardScaler()
#         df_scaled = scaler.fit_transform(df)
#         reconstructed = autoencoder.predict(df_scaled, verbose=0)
#         reconstruction_errors = np.mean(np.square(df_scaled - reconstructed), axis=1)
        
#         results.append({
#             'run': i+1,
#             'seed': 42 + i,
#             'mean_reconstruction_error': np.mean(reconstruction_errors),
#             'std_reconstruction_error': np.std(reconstruction_errors),
#         })
    
#     # Print summary
#     print(f"\n{'='*60}")
#     print(f"📊 EXPERIMENT SUMMARY")
#     print(f"{'='*60}")
#     for result in results:
#         print(f"Run {result['run']} (seed {result['seed']}): Error = {result['mean_reconstruction_error']:.4f}")
    
#     return results
# run_multiple_experiments()


