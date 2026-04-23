import Shiv_Shankar_autoencoder
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import ruptures as rpt
from sklearn.mixture import GaussianMixture
from scipy import stats



def get_decimal_years(date_series):
    return (date_series.dt.year + 
            (date_series.dt.dayofyear - 1) / 365.25)

def detect_change_points_multiple_methods(time_series_data, c_df):
    """
    Detect change points using multiple methods
    """
    change_points_results = {}
    # Method 1: PELT (Pruned Exact Linear Time)
    algo_pelt = rpt.Pelt(model="rbf", min_size=10, jump=1).fit(time_series_data)
    pelt_change_points = algo_pelt.predict(pen=10)
    change_points_results['PELT'] = pelt_change_points[:-1]  # Remove last point (end of series)
    
    # Method 2: Binary Segmentation
    algo_binseg = rpt.Binseg(model="l2", min_size=10, jump=1).fit(time_series_data)
    binseg_change_points = algo_binseg.predict(n_bkps=5)
    change_points_results['BinSeg'] = binseg_change_points[:-1]
    
    # Method 3: Window-based detection
    algo_window = rpt.Window(width=50, model="l2", min_size=10, jump=1).fit(time_series_data)
    window_change_points = algo_window.predict(n_bkps=5)
    change_points_results['Window'] = window_change_points[:-1]
    
    # Method 4: Kernel change point detection
    algo_kernel = rpt.KernelCPD(kernel="linear", min_size=10, jump=1).fit(time_series_data)
    kernel_change_points = algo_kernel.predict(n_bkps=4)
    change_points_results['Kernel'] = kernel_change_points[:-1]
    
    # Convert indices to dates
    dates = c_df['Date'].values
    change_points_dates = {}
    
    for method, cp_indices in change_points_results.items():
        cp_dates = [dates[min(idx, len(dates)-1)] for idx in cp_indices if idx < len(dates)]
        change_points_dates[method] = cp_dates
        print(f"{method} detected change points at dates: {cp_dates}")
    
    return change_points_results, change_points_dates

def visualize_change_points(change_points_latent,c_df):
    """
    Visualize LATENT-BASED change points on original climate variables
    Focus: Major climate regime transitions only
    """
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    variables = ['Temp_Anomaly_°C', 'CO2_in_ppm', 'Precip_in_mm', 
                'Nino34_in_°C', 'TSI_Wm2', 'Volcanic_Global']
    
    for i, var in enumerate(variables):
        ax = axes[i]
        data = c_df[var]
        
        # Plot time series
        ax.plot(c_df['Date'], data, 'b-', alpha=0.7, linewidth=1.5, label='Climate Data')
        
        # Add LATENT-BASED change points (Major climate regime shifts)
        colors = ['red', 'yellow', 'blue', 'cyan']
        methods = ['PELT', 'BinSeg', 'Window', 'Kernel']
        
        for j, method in enumerate(methods):
            if method in change_points_latent:
                for cp_date in change_points_latent[method]:
                    ax.axvline(x=pd.to_datetime(cp_date), color=colors[j], linestyle='-', 
                              alpha=0.8, linewidth=2,
                              label=f'{method}' if cp_date == change_points_latent[method][0] else "")
        
        ax.set_title(f'{var}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year',loc='right')
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('CLIMATE REGIME ANALYSIS: Latent-Based Change Points\n' + 
                 'Focus: System-Wide Climate Transitions', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_change_points_PELT(change_points_latent,c_df):
    fig, axes = plt.subplots(3, 2, figsize=(20, 15))
    axes = axes.flatten()
    
    variables = ['Temp_Anomaly_°C', 'CO2_in_ppm', 'Precip_in_mm', 
                'Nino34_in_°C', 'TSI_Wm2', 'Volcanic_Global']
    
    for i, var in enumerate(variables):
        ax = axes[i]
        data = c_df[var]
        
        # Plot time series
        ax.plot(c_df['Date'], data, 'b-', alpha=0.7, linewidth=1.5, label='Climate Data')
        for cp_date in change_points_latent['PELT']:
            ax.axvline(x=pd.to_datetime(cp_date), color='r', linestyle='-', alpha=0.8, linewidth=2,label='PELT' if cp_date == change_points_latent['PELT'][0] else "")
        
        ax.set_title(f'{var}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Year',loc='right')
        ax.set_ylabel(var)
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.suptitle('CLIMATE REGIME ANALYSIS THROUGH PELT', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def analyze_climate_regimes(latent_df, change_points_dates, c_df):
    """
    Analyze climate regimes between change points using PELT (most reliable)
    """
    # Use PELT as primary method (most reliable for climate regimes)
    primary_method = 'PELT'
    visualize_change_points_PELT(change_points_dates,c_df)
    # Extract change point years based on PELT
    cp_dates = change_points_dates[primary_method]
    cp_years = [pd.to_datetime(date).year for date in cp_dates]
    cp_years = sorted(cp_years)
    
    # Add start and end years for complete regime analysis
    start_year = c_df['Year'].min()
    end_year = c_df['Year'].max()
    all_years = [start_year] + cp_years + [end_year]
    
    print(f"\n📅 Change Point Years: {cp_years}")
    print(f"🎯 Climate Regime Periods: {[(all_years[i], all_years[i+1]) for i in range(len(all_years)-1)]}")
    
    # Analyze each regime
    regime_analysis = []
    
    for i in range(len(all_years) - 1):
        start_year_regime = all_years[i]
        end_year_regime = all_years[i + 1]
        
        # Get data for this regime
        regime_mask = (c_df['Year'] >= start_year_regime) & (c_df['Year'] < end_year_regime)
        regime_data = c_df[regime_mask]
        
        # Match latent data (assuming latent_df has Year column)
        if 'Year' not in latent_df.columns:
            latent_df['Year'] = pd.to_datetime(latent_df['Date']).dt.year
        
        regime_latent = latent_df[latent_df['Year'].isin(regime_data['Year'])]

        decimal_years = get_decimal_years(regime_data['Date'])
        
        # Calculate comprehensive statistics
        regime_stats = {
            'regime': i + 1,
            'period': f"{start_year_regime}-{end_year_regime}",
            'n_years': end_year_regime - start_year_regime,
            'start_year': start_year_regime,
            'end_year': end_year_regime,
            
            # Climate variable means
            'avg_temp_anomaly': regime_data['Temp_Anomaly_°C'].mean(),
            'avg_co2': regime_data['CO2_in_ppm'].mean(),
            'avg_precip': regime_data['Precip_in_mm'].mean(),
            'avg_nino34': regime_data['Nino34_in_°C'].mean(),
            'avg_volcanic': regime_data['Volcanic_Global'].mean(),
            'avg_tsi': regime_data['TSI_Wm2'].mean(),
            
            # Latent space means
            'avg_latent_1': regime_latent['Latent_1'].mean(),
            'avg_latent_2': regime_latent['Latent_2'].mean(),
            'avg_latent_3': regime_latent['Latent_3'].mean(),

            # Trends (slopes)
            'temp_trend': np.polyfit(decimal_years, regime_data['Temp_Anomaly_°C'],1)[0] if len(regime_data) > 1 else 0,
            'co2_trend': np.polyfit(decimal_years, regime_data['CO2_in_ppm'], 1)[0] if len(regime_data) > 1 else 0,
            'nino_trend' : np.polyfit(decimal_years, regime_data['Nino34_in_°C'], 1)[0] if len(regime_data) > 1 else 0,
            'precip_trend' : np.polyfit(decimal_years, regime_data['Precip_in_mm'], 1)[0] if len(regime_data) > 1 else 0,
            
            # Variability measures
            'temp_std': regime_data['Temp_Anomaly_°C'].std(),
            'co2_std': regime_data['CO2_in_ppm'].std(),
        }
        regime_analysis.append(regime_stats)

    # Create regime comparison Dataframe
    regime_df = pd.DataFrame(regime_analysis)

    # Visualize regime characteristics
    plt.figure(1,figsize=(15, 10))

    # Temperature and CO2 by regime
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(regime_df['regime'], regime_df['avg_temp_anomaly'], alpha=0.7, color='red')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_title('Average Temperature Anomaly by Regime')
    ax1.set_xlabel('Regime')
    ax1.set_ylabel('Temperature Anomaly (°C)')
    ax1.bar_label(bars1, fmt='%.4f')
    
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(regime_df['regime'], regime_df['avg_co2'], alpha=0.7, color='blue')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('Average CO2 Concentration by Regime')
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('CO2 (ppm)')
    ax2.bar_label(bars2, fmt='%.4f')
    
    # Latent features by regime
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(regime_df['regime'], regime_df['avg_precip'], alpha=0.7, color='green')
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_title('Average Precipitation by Regime')
    ax3.set_xlabel('Regime')
    ax3.set_ylabel('Precipitation (mm)')
    ax3.bar_label(bars3, fmt='%.4f')

    # Trends by regime
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.bar(regime_df['regime'], regime_df['avg_nino34'], alpha=0.7, color='cyan')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)    
    ax4.set_title('Average Nino34 by Regime')
    ax4.set_xlabel('Regime')
    ax4.set_ylabel('Nino34 (°C)')
    ax4.bar_label(bars4, fmt='%.4f')
    
    ax5 = plt.subplot(2, 3, 5)
    bars5 = ax5.bar(regime_df['regime'], regime_df['avg_volcanic'], alpha=0.7, color='purple')
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax5.set_title('Average Volcanic Global by Regime')
    ax5.set_xlabel('Regime')
    ax5.set_ylabel('Volcanic Global')
    ax5.bar_label(bars5, fmt='%.4f')
    
    # Regime duration
    ax6 = plt.subplot(2, 3, 6)
    bars6 = ax6.bar(regime_df['regime'], regime_df['avg_tsi'], alpha=0.7, color='magenta')
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax6.set_title('Average TSI by Regime')
    ax6.set_xlabel('Regime')
    ax6.set_ylabel('TSI (Wm2)')
    ax6.bar_label(bars6, fmt='%.4f')
    
    plt.suptitle('CLIMATE REGIME COMPARISON-Part 1', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    plt.figure(2,figsize=(15, 10))

    # Temperature and CO2 by regime
    ax1 = plt.subplot(2, 3, 1)
    bars1 = ax1.bar(regime_df['regime'], regime_df['temp_trend'], alpha=0.7, color='orange')
    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax1.set_title('Temperature Trend by Regime')
    ax1.set_xlabel('Regime')
    ax1.set_ylabel('Temperature Trend (°C/year)')
    ax1.bar_label(bars1, fmt='%.4f')
    
    ax2 = plt.subplot(2, 3, 2)
    bars2 = ax2.bar(regime_df['regime'], regime_df['co2_trend'], alpha=0.7, color='green')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax2.set_title('CO2 Trend by Regime')
    ax2.set_xlabel('Regime')
    ax2.set_ylabel('CO2 Trend (ppm/year)')
    ax2.bar_label(bars2, fmt='%.4f')
    
   
    ax3 = plt.subplot(2, 3, 3)
    bars3 = ax3.bar(regime_df['regime'], regime_df['nino_trend'], alpha=0.7, color='magenta')    
    ax3.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax3.set_title('Nino34 Trend by Regime')
    ax3.set_xlabel('Regime')
    ax3.set_ylabel('Nino34 Trend (°C/year)')
    ax3.bar_label(bars3, fmt='%.4f')

    # Trends by regime
    ax4 = plt.subplot(2, 3, 4)
    bars4 = ax4.bar(regime_df['regime'], regime_df['precip_trend'], alpha=0.7, color='blue')
    ax4.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax4.set_title('Precipitation Trend by Regime')
    ax4.set_xlabel('Regime')
    ax4.set_ylabel('Precipitation Trend (mm/year)')
    ax4.bar_label(bars4, fmt='%.4f')
    
    # Latent features by regime
    ax5 = plt.subplot(2, 3, 5)
    width = 0.25
    x = np.arange(len(regime_df))
    bars1 = ax5.bar(x - width, regime_df['avg_latent_1'], width, label='Latent 1', alpha=0.7)
    bars2 = ax5.bar(x, regime_df['avg_latent_2'], width, label='Latent 2', alpha=0.7)
    bars3 = ax5.bar(x + width, regime_df['avg_latent_3'], width, label='Latent 3', alpha=0.7)
    ax5.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax5.set_title('Latent Features by Regime')
    ax5.set_xlabel('Regime')
    ax5.set_ylabel('Latent Value')
    ax5.legend()
    ax5.bar_label(bars1, fmt='%.4f')
    ax5.bar_label(bars2, fmt='%.4f')
    ax5.bar_label(bars3, fmt='%.4f')
    
    # Regime duration
    ax6 = plt.subplot(2, 3, 6)
    bars6 = ax6.bar(regime_df['regime'], regime_df['n_years'], alpha=0.7, color='purple')
    ax6.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax6.set_title('Regime Duration')
    ax6.set_xlabel('Regime')
    ax6.set_ylabel('Years')
    ax6.bar_label(bars6, fmt='%.4f')
    
    plt.suptitle('CLIMATE REGIME COMPARISON-Part 2', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return regime_df

def anomaly_detection_autoencoder(autoencoder,df,df_scaled, c_df, threshold_percentile=95):
    """
    Use autoencoder reconstruction error for anomaly detection
    """
    # Get reconstructions
    reconstructions = autoencoder.predict(df_scaled)
    
    # Calculate reconstruction error
    mse = np.mean(np.power(df_scaled - reconstructions, 2), axis=1)
    
    # Set threshold for anomalies
    threshold = np.percentile(mse, threshold_percentile)
    anomalies = mse > threshold
    
    print(f"Detected {np.sum(anomalies)} anomalies ({np.sum(anomalies)/len(anomalies)*100:.4f}% of data)")
    print(f"Reconstruction error threshold: {threshold:.4f}")
    
    # Identify anomalous years and months
    anomaly_dates = c_df[anomalies]['Date'].values
    anomaly_data = c_df[anomalies]

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Reconstruction error over time
    axes[0,0].plot(c_df['Date'], mse, 'b-', alpha=0.7)
    axes[0,0].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold_percentile}th percentile)')
    axes[0,0].scatter(c_df[anomalies]['Date'], mse[anomalies], color='red', s=50, alpha=0.8, label='Anomalies')
    axes[0,0].set_title('A) Reconstruction Error Over Time')
    axes[0,0].set_xlabel('Date')
    axes[0,0].set_ylabel('Reconstruction Error (MSE)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Reconstruction error distribution
    axes[1,0].hist(mse, bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].axvline(x=threshold, color='r', linestyle='--', label=f'Threshold = {threshold:.4f}')
    axes[1,0].set_title('C) Distribution of Reconstruction Errors')
    axes[1,0].set_xlabel('Reconstruction Error (MSE)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Feature importance for anomalies
    feature_errors = np.mean(np.abs(df_scaled[anomalies] - reconstructions[anomalies]), axis=0)
    feature_names = df.columns
    
    bars = axes[0,1].bar(range(len(feature_names)), feature_errors, alpha=0.7)
    axes[0,1].set_title('B) Average Feature Reconstruction Error for Anomalies')
    axes[0,1].set_xlabel('Features')
    axes[0,1].set_ylabel('Average Absolute Error')
    axes[0,1].set_xticks(range(len(feature_names)))
    axes[0,1].set_xticklabels(feature_names, rotation=45)
    axes[0,1].bar_label(bars, fmt='%.4f')
    axes[0,1].grid(True, alpha=0.3)
    
    plt.suptitle('DETECTION OF ANOMALIES', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return anomalies, mse, threshold

def visualize_anomaly_timeline(c_df, anomalies, mse):
    """Create timeline showing anomalies with their causes"""
    
    # Define major climate events
    climate_events = {
        '1982-1983': {'name': 'El Chichón Eruption', 'color': 'red', 'type': 'Volcanic'},
        '1991-1992': {'name': 'Mt. Pinatubo Eruption', 'color': 'darkred', 'type': 'Volcanic'},
        '1982-1983_ENSO': {'name': '1982-83 Super El Niño', 'color': 'orange', 'type': 'ENSO'},
        '1988-1989': {'name': '1988-89 La Niña', 'color': 'blue', 'type': 'ENSO'},
        '1997-1998': {'name': '1997-98 El Niño', 'color': 'orange', 'type': 'ENSO'},
        '2000s': {'name': 'Anthropogenic Warming', 'color': 'purple', 'type': 'Warming'}
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    
    # Plot 1: Temperature with anomalies
    axes[0].plot(c_df['Date'], c_df['Temp_Anomaly_°C'], 'b-', alpha=0.7, linewidth=1)
    axes[0].scatter(c_df[anomalies]['Date'], c_df[anomalies]['Temp_Anomaly_°C'], 
                   color='red', s=60, alpha=0.8, zorder=5)
    
    # Add event annotations
    axes[0].axvspan(pd.to_datetime('1982-01'), pd.to_datetime('1984-01'), 
                   alpha=0.2, color='red', label='El Chichón Period')
    axes[0].axvspan(pd.to_datetime('1991-01'), pd.to_datetime('1993-01'), 
                   alpha=0.2, color='darkred', label='Pinatubo Period')
    axes[0].axvspan(pd.to_datetime('1997-01'), pd.to_datetime('1998-12'), 
                   alpha=0.2, color='orange', label='1997-98 El Niño')
    
    axes[0].set_title('A) Temperature Anomalies with Climate Events', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Temperature Anomaly (°C)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Volcanic forcing with anomalies
    axes[1].plot(c_df['Date'], c_df['Volcanic_Global'], 'g-', alpha=0.7, linewidth=1)
    axes[1].scatter(c_df[anomalies]['Date'], c_df[anomalies]['Volcanic_Global'], 
                   color='red', s=60, alpha=0.8, zorder=5)
    axes[1].set_title('B) Volcanic Forcing with Detected Anomalies', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Volcanic Global Forcing')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: ENSO with anomalies
    axes[2].plot(c_df['Date'], c_df['Nino34_in_°C'], 'm-', alpha=0.7, linewidth=1)
    axes[2].scatter(c_df[anomalies]['Date'], c_df[anomalies]['Nino34_in_°C'], 
                   color='red', s=60, alpha=0.8, zorder=5)
    axes[2].axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, label='El Niño Threshold')
    axes[2].axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5, label='La Niña Threshold')
    axes[2].set_title('C) ENSO (Niño 3.4) with Detected Anomalies', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Niño 3.4 Index (°C)')
    axes[2].set_xlabel('Year')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.suptitle('CLIMATE ANOMALIES: Causes and Detection\n', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_anomaly_causes(c_df, anomalies):
    """Create matrix showing anomaly causes"""
    
    # Extract anomaly data
    anomaly_data = c_df[anomalies].copy()
    
    # Categorize anomalies by cause
    def categorize_anomaly(row):
        volcanic = row['Volcanic_Global']
        nino_abs = abs(row['Nino34_in_°C'])  
        year = row['Date'].year
        temp = row['Temp_Anomaly_°C']
        
        if volcanic > 0.05:
            return 'Volcanic'
        elif nino_abs > 1.5:  # Strong ENSO (El Niño OR La Niña)
            return 'Strong ENSO'
        elif nino_abs > 0.5:  # 🔧 Moderate ENSO  
            return 'Moderate ENSO'
        elif year >= 2000 and temp > 0.5:
            return 'Anthropogenic'
        else:
            return 'Natural Variability'
    
    anomaly_data['Cause'] = anomaly_data.apply(categorize_anomaly, axis=1)
    
    # Create cause-effect visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Cause distribution
    cause_counts = anomaly_data['Cause'].value_counts()
    axes[0,0].pie(cause_counts.values, labels=cause_counts.index, autopct='%1.1f%%',
                 colors=['red', 'orange', 'blue', 'purple', 'green'])
    axes[0,0].set_title('A) Anomaly Causes Distribution')
    
    # Temporal distribution
    anomaly_data['Decade'] = (anomaly_data['Date'].dt.year // 10) * 10
    decade_counts = anomaly_data.groupby(['Decade', 'Cause']).size().unstack(fill_value=0)
    decade_counts.plot(kind='bar', stacked=True, ax=axes[0,1], 
                      color=['red', 'orange', 'blue', 'purple', 'green'])
    
    # ADD BAR LABELS:
    for container in axes[0,1].containers:
        axes[0,1].bar_label(container, label_type='center', fontsize=8, 
                            fmt='%g', color='white', weight='bold')
        
    axes[0,1].set_title('B) Anomalies by Decade and Cause')
    axes[0,1].set_xlabel('Decade',loc='right')
    axes[0,1].set_ylabel('Number of Anomalies')
    axes[0,1].legend(title='Cause')
    
    # Intensity vs Cause
    for cause in anomaly_data['Cause'].unique():
        cause_data = anomaly_data[anomaly_data['Cause'] == cause]
        axes[1,0].scatter(cause_data['Date'], cause_data['Temp_Anomaly_°C'], 
                         label=cause, s=60, alpha=0.7)
    axes[1,0].set_title('C) Temperature Anomaly Intensity by Cause')
    axes[1,0].set_xlabel('Year')
    axes[1,0].set_ylabel('Temperature Anomaly (°C)')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Multi-variable impact
    causes = anomaly_data['Cause'].unique()
    variables = ['Temp_Anomaly_°C', 'CO2_in_ppm', 'Nino34_in_°C', 'Volcanic_Global']

    # Create matrix of average impacts
    impact_matrix = np.zeros((len(causes), len(variables)))
    for i, cause in enumerate(causes):
        cause_data = anomaly_data[anomaly_data['Cause'] == cause]
        for j, var in enumerate(variables):
            impact_matrix[i, j] = cause_data[var].mean()

    im = axes[1,1].imshow(impact_matrix, cmap='RdBu_r', aspect='auto')
    axes[1,1].set_xticks(range(len(variables)))
    axes[1,1].set_xticklabels(variables, rotation=45, ha='right', fontsize=10)
    axes[1,1].set_yticks(range(len(causes)))
    axes[1,1].set_yticklabels(causes, fontsize=10)
    axes[1,1].set_title('D) Average Impact by Cause', fontsize=14, fontweight='bold')

    # Annotate cells with values
    for i in range(len(causes)):
        for j in range(len(variables)):
            axes[1,1].text(j, i, f"{impact_matrix[i, j]:.2f}",
                    ha='center', va='center',
                    color='white',
                    fontsize=8)

    # Add colorbar
    plt.colorbar(im, ax=axes[1,1])
    
    plt.suptitle('CLIMATE ANOMALY CAUSE ANALYSIS', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return anomaly_data

def create_anomaly_summary_table(c_df, anomalies):
    """Create detailed anomaly summary table"""
    
    anomaly_data = c_df[anomalies].copy()
    
    # Add cause classification
    def get_primary_cause(row):
        causes = []
        volcanic = row['Volcanic_Global'] 
        nino_abs = abs(row['Nino34_in_°C'])  
        nino_val = row['Nino34_in_°C']       
        
        if volcanic > 0.05:
            causes.append(f"Volcanic ({volcanic:.3f})")
        if nino_abs > 1.5:  # Check absolute value
            causes.append(f"Strong ENSO ({nino_val:.1f}°C)")  # Display original value
        elif nino_abs > 0.5:  # Check absolute value
            causes.append(f"Moderate ENSO ({nino_val:.1f}°C)")  # Display original value
        if row['Date'].year >= 2000 and row['Temp_Anomaly_°C'] > 0.5:
            causes.append("Anthropogenic Warming")
        
        return " + ".join(causes) if causes else "Natural Variability"
    
    anomaly_data['Primary_Cause'] = anomaly_data.apply(get_primary_cause, axis=1)
    
    # Create summary table
    summary_cols = ['Date', 'Temp_Anomaly_°C', 'CO2_in_ppm', 'Nino34_in_°C', 
                   'Volcanic_Global', 'Primary_Cause']
    
    summary_table = anomaly_data[summary_cols].copy()
    summary_table['Year-Month'] = summary_table['Date'].dt.strftime('%Y-%m')
    
    print("🌍 CLIMATE ANOMALY SUMMARY TABLE")
    print("="*80)
    print(f"{'Year-Month':<10} {'Temp(°C)':<8} {'ENSO(°C)':<8} {'Volcanic':<9} {'Primary Cause':<30}")
    print("-"*80)
    
    for _, row in summary_table.iterrows():
        print(f"{row['Year-Month']:<10} {row['Temp_Anomaly_°C']:>7.2f} "
              f"{row['Nino34_in_°C']:>7.2f} {row['Volcanic_Global']:>8.3f} "
              f"{row['Primary_Cause']:<30}")
    
    return summary_table

def comprehensive_anomaly_visualization(autoencoder, df_scaled, c_df,anomalies):
    """Complete anomaly visualization suite"""

    print(f"🔍 COMPREHENSIVE ANOMALY ANALYSIS")
    print(f"Detected {np.sum(anomalies)} anomalies ({np.sum(anomalies)/len(anomalies)*100:.1f}% of data)")
    print("="*60)
    
    # 1. Timeline visualization
    visualize_anomaly_timeline(c_df, anomalies, mse)
    
    # 2. Cause-effect analysis
    visualize_anomaly_causes(c_df, anomalies)
    
    # 3. Summary table
    summary_table = create_anomaly_summary_table(c_df, anomalies)
    
    return summary_table


c_df = pd.read_csv('Main_df.csv')
c_df['Date'] = pd.to_datetime(c_df['Date'])
c_df['Decade'] = (c_df['Date'].dt.year // 10) * 10
c_df['Year'] = c_df['Date'].dt.year

df = c_df[['Temp_Anomaly_°C', 'CO2_in_ppm', 'Precip_in_mm', 'Nino34_in_°C', 'Volcanic_Global', 'TSI_Wm2']]
    
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

autoencoder, encoder, decoder, latent_df, latent_features = Shiv_Shankar_autoencoder.main()
change_points_results,change_points_dates = detect_change_points_multiple_methods(latent_features, c_df)
visualize_change_points(change_points_dates,c_df)
regime_df = analyze_climate_regimes(latent_df,change_points_dates, c_df)
print('\n📊 Climatic Regime Characteristics:')
print(regime_df[['avg_temp_anomaly','avg_co2','avg_nino34','avg_precip']].round(4))
print(regime_df[['avg_volcanic','avg_tsi','avg_latent_1','avg_latent_2','avg_latent_3']].round(4))
print(regime_df[['temp_trend','co2_trend','nino_trend','precip_trend']].round(4))
anomalies, mse, threshold = anomaly_detection_autoencoder(autoencoder,df,df_scaled, c_df)
summary_table = comprehensive_anomaly_visualization(autoencoder, df_scaled, c_df,anomalies)
print('SUMMARY TABLE')
print(summary_table)
# print("\n🚀 STARTING LSTM+AUTOENCODER FORECASTING ANALYSIS...")


