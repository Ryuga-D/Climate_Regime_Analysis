import pandas as pd
import numpy as np
import xarray as xr
import glob

# read in the Global Precipitation Climatology Centre (GPCC) Full Data on a 0.5° grid.
# The amount of rainfall/snowfall (precipitation) over each 0.5° × 0.5° grid cell on Earth.
# A spatial grid where each grid cell is 50 km to 50 km square on the Earth’s surface roughly
# Source : Deutscher Wetterdienst (DWD), Germany

ds = xr.open_dataset('./DATASETS/Global_monthly_precipitation_full_data_v7_05.nc.gz')
df_prep = ds['p'].mean(dim = ['lat','lon']).to_dataframe().reset_index()
df_prep.rename(columns={'time':'Date','p':'Precip_in_mm'},inplace=True)
df_prep['Date'] = pd.to_datetime(df_prep['Date'])
# print(df_prep)

# read in the Global Temperature Anomally dataset from NASA GISTEMP Global Land-Ocean Temperature Index
# Source : NASA Goddard Institute for Space Studies (GISS)
# How much warmer or cooler the Earth was compared to the average temperature during 1951–1980

ds = pd.read_csv('./DATASETS/NASA_Global_Temp_monthly.csv')
ds = ds[ds['Source']=='GISTEMP'] # Row filtering to include only rows with source as GISTEMP
ds['Year'] = pd.to_datetime(ds['Year'])
df_temp = ds[['Year','Mean']].rename(columns={'Mean':'Temp_Anomaly_°C','Year':'Date'}).reset_index(drop =True)
# print(df_temp.head())

# read in the NOAA ESRL (Earth System Research Laboratories) Co2 data
# Source : The NOAA CO₂ dataset we're using is based on measurements from Mauna Loa Observatory in Hawaii.
# The Observatory is high on a volcano and far from cities, so it gets clean, well-mixed air that reflects global background CO₂.

ds = pd.read_csv('./DATASETS/co2_monthly_mm_mlo.csv',comment = '#',header = 0)
ds.columns = ['Year','Month','Decimal Date','Average','Deseasonalized','NDays','StDev','Uncertainty']
ds['Date'] = pd.to_datetime(dict(year=ds.Year, month=ds.Month, day=1)) # Create a date column from year and month and set the day to 1
df_co2 = ds[['Date', 'Deseasonalized']].rename(columns={'Deseasonalized': 'CO2_in_ppm'})
# print(df_co2.head())

# Merge all three datasets based on common date 
df_merged = pd.merge(df_temp, df_co2, on='Date', how='inner')
df_merged = pd.merge(df_merged, df_prep, on='Date', how='inner')
# print(df_merged.head())

# Source :  NOAA's Climate Prediction Center (CPC)
# Niño 3.4 Sea Surface Temperature (SST) Anomaly Index
# Each value is a monthly sea surface temperature anomaly (°C) where anomaly = actual temperature – long-term average baseline
# Here, positive values → El Niño (ocean is warmer than normal)
# Negative values → La Niña (ocean is cooler than normal)
# 3.4 is a central portion of the equatorial Pacific that plays a key role in the ENSO (El Niño–Southern Oscillation) phenomenon.
# The Niño 3.4 index is a global climate signal — it influences rainfall, droughts, storms, and temperature trends across the globe.

df_nino = pd.read_csv('./DATASETS/nino34.long.anom.csv', skiprows=1)
# The first column appears to be "Date," (with comma) and second is "NINA34"
# Let's clean up the column names and data
df_nino.columns = ['Date', 'Nino34'] + [f'extra_{i}' for i in range(len(df_nino.columns)-2)]
# Clean the Date column (remove trailing commas)
df_nino['Date'] = df_nino['Date'].astype(str).str.rstrip(',')
df_nino['Date'] = pd.to_datetime(df_nino['Date'])
# Keep only Date and Nino34 columns
df_nino_long = df_nino[['Date', 'Nino34']].copy()
# Replace missing values (-99.99) with NaN beacuse -99.99 is a known placeholder used in NOAA data for missing values.
df_nino_long['Nino34'] = pd.to_numeric(df_nino_long['Nino34'], errors='coerce')
df_nino_long['Nino34'] = df_nino_long['Nino34'].replace(-99.99, np.nan)
# print("Processed NINO data:")
# print(df_nino_long.head())

#. Merge the NINO data with the main dataset based on the "Date" column
df_merged_final = df_merged.merge(df_nino_long, on='Date', how='left')
df_merged_final.rename(columns={'Nino34': 'Nino34_in_°C'}, inplace=True)
# print(df_merged_final.tail())
# print(df_merged_final.isna().sum()) # after printing, we get no missing values in the NINO data and other columns
# print(f"Date range: {df_merged_final['Date'].min()} to {df_merged_final['Date'].max()}")


# Source: NASA Goddard Institute for Space Studies (GISS)
# NASA GISS Stratospheric Aerosol Optical Thickness (SAOD), global monthly means
# The SAOD dataset is a monthly mean of the optical thickness of aerosols in the stratosphere globally.
# AOD (Aerosol Optical Depth) quantifies how much sunlight is blocked by aerosols in the stratosphere.

df_volcano = pd.read_csv('./DATASETS/Volcanic.csv')
df_volcano.columns = ['YearFrac', 'Volcanic_Global', 'Volcanic_NH', 'Volcanic_SH']
# Convert fractional year (e.g., 1958.042) to datetime
df_volcano['Year'] = df_volcano['YearFrac'].astype(int)
df_volcano['MonthFrac'] = (df_volcano['YearFrac'] - df_volcano['Year']) * 12 + 1
df_volcano['Month'] = df_volcano['MonthFrac'].round().astype(int)
# Ensure month values are valid (1-12) - just as a safety check
df_volcano['Month'] = df_volcano['Month'].clip(1, 12)
df_volcano['Date'] = pd.to_datetime(dict(year=df_volcano['Year'], month=df_volcano['Month'], day=1))
# Keep only Date and Global AOD
df_volcano = df_volcano[['Date', 'Volcanic_Global']]
# drop duplicates and group by date to get mean values of column data
df_volcano = df_volcano.drop_duplicates().groupby('Date').mean(numeric_only=True).reset_index() 
# print(df_volcano)
# print(df_volcano.shape)


# Merge with the main dataset
df_final = df_merged_final.merge(df_volcano, on='Date', how='left')
df_final['Volcanic_Global'] = df_final['Volcanic_Global'].fillna(0)
# print("Final merged dataset:")
# print(df_final.head())
# print(df_final.shape)


# Total Solar Irradiance (TSI) – Monthly Average measured in watts per square meter (W/m²).
# Source: NOAA National Centers for Environmental Information (NCEI)
#TSI represents the amount of solar energy received at the top of Earth's atmosphere from the Sun per unit area. 
# It is a critical climate variable because solar radiation is the primary energy source driving Earth's climate system

tsi_files = glob.glob("./DATASETS/Solar*.nc")  
all_tsi_data = []
for file in tsi_files:
    ds = xr.open_dataset(file)
    df = ds.to_dataframe().reset_index()
    df = df[['time', 'TSI']] 
    df.columns = ['Date', 'TSI_Wm2']
    df['Date'] = pd.to_datetime(df['Date'])
    df.drop_duplicates(inplace=True)
    df['Date'] = df['Date'].apply(lambda x: x.replace(day=1))
    all_tsi_data.append(df)

combined_tsi = pd.concat(all_tsi_data, ignore_index=True)
combined_tsi = combined_tsi.drop_duplicates(subset=['Date']).sort_values('Date').reset_index(drop=True)
Ultimate_df = df_final.merge(combined_tsi, on='Date', how='left')
print("Ultimate merged dataset:")
# print(Ultimate_df.head())
# Ultimate_df.to_csv('Main1_df.csv', index=False)
# print(Ultimate_df.isna().sum())
