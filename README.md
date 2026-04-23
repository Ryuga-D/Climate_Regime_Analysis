<div align="center">
  
# 🌍 Climate Regime & Anomaly Detection Using Deep Learning

**A state-of-the-art machine learning framework for analyzing global climate patterns, detecting regime shifts, and identifying extreme climate anomalies.**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://tensorflow.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-yellow.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

</div>

---

## 📖 Overview

This repository contains a comprehensive deep learning pipeline to analyze multi-decadal climate data. By leveraging Autoencoders for feature extraction and advanced time-series forecasting models (LSTM, BiLSTM, Seq2Seq), this project identifies critical climate change points and anomalies over time.

It processes key climate indicators including:
- 🌡️ **Temperature Anomalies (°C)**
- 🏭 **CO2 Concentrations (ppm)**
- 🌧️ **Precipitation (mm)**
- 🌊 **ENSO (Niño 3.4 Index)**
- 🌋 **Volcanic Forcing**
- ☀️ **Total Solar Irradiance (TSI)**

---

## 🎯 Key Features

### 1. 🔄 Unsupervised Feature Extraction
Uses deep **Autoencoders** to compress high-dimensional climate data into a latent space, capturing underlying nonlinear relationships between atmospheric, oceanic, and solar variables.

### 2. 📉 Climate Regime Change Detection
Employs multiple algorithmic methods to detect fundamental shifts in climate behavior:
- **PELT (Pruned Exact Linear Time)** - Primary reliable method
- **Binary Segmentation**
- **Window-based Detection**
- **Kernel Change Point Detection**

*Identified Regimes:*
- **1958-1977**: Natural variability & La Niña bias.
- **1977-1994**: Regime shift, El Niño tendency, peak volcanic activity.
- **1994-2005**: Anthropogenic warming acceleration.
- **2005-2013**: Contemporary warming, high precipitation, minimum TSI.

### 3. 🚨 Extreme Anomaly Detection
Uses the Autoencoder's reconstruction error (MSE) to pinpoint historical climate anomalies with remarkable accuracy:
- **Major Volcanic Eruptions**: El Chichón (1982), Mt. Pinatubo (1991)
- **Extreme ENSO Events**: 1982-83 Super El Niño, 1997-98 El Niño, Strong La Niña (1988-89)
- **Anthropogenic Warming Spikes**: Early 2000s & 2010

### 4. 🔮 Advanced Time-Series Forecasting
Multiple state-of-the-art models for future climate prediction:
- **Trend-Focused Ensemble Model**: A sophisticated hybrid approach combining multiple architectures using a custom weighted ensemble based on trend accuracy.
- **BiLSTM**: Bidirectional context understanding.
- **CNN-LSTM**: Spatial and temporal pattern recognition.
- **Seq2Seq Encoder-Decoder**: Long-term multi-step forecasting.
- **SARIMA**: Traditional statistical baseline for seasonal adjustments.

---

## 📂 Repository Structure

```text
📦 Climate-Regime-Analysis
 ┣ 📜 Main_df.csv                       # Primary climate dataset
 ┣ 📜 Main_Test.csv                     # Testing dataset
 ┣ 📜 Shiv_Shankar_main.py              # Main pipeline & visualization suite
 ┣ 📜 Shiv_Shankar_autoencoder.py       # Deep Autoencoder architecture
 ┣ 📜 Shankar_Model.py                  # Core modeling utilities
 ┣ 📜 Testing1_bilstm.py                # BiLSTM forecasting experiments
 ┣ 📜 Testing2_lstm_cnn.py              # Hybrid LSTM-CNN modeling
 ┣ 📜 Testing3_seq2seq_encoder_decoder.py # Seq2Seq forecasting
 ┣ 📜 Information.py                    # Metadata and scientific insights
 ┗ 📜 README.md                         # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites
Ensure you have Python 3.8+ installed. The project relies on standard data science and deep learning libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow ruptures scipy
```

### Running the Pipeline

1. **Run the core analysis and visualization suite:**
   ```bash
   python Shiv_Shankar_main.py
   ```
2. **Train and evaluate specific forecasting models:**
   ```bash
   python Testing1_bilstm.py
   # or
   python Testing2_lstm_cnn.py
   ```

---

## 📊 Scientific Insights & Results

The system successfully validates known climate physics purely from data:
1. **Volcanic-Climate Coupling**: The model organically detected cooling effects following major eruptions (e.g., Pinatubo).
2. **Pacific Decadal Oscillation (PDO)**: The 1977 Pacific regime shift from La Niña to El Niño dominance is clearly identified.
3. **Anthropogenic Signal**: Detected an exponentially increasing frequency of warm anomalies in the 21st century, decoupling from natural forcings (TSI, volcanoes).

---

<div align="center">
  <i>"Understanding our past climate to better forecast our future."</i>
</div>
