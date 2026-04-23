import pandas as pd
import numpy as np
import xarray as xr
import glob

ds = xr.open_dataset('./DATASETS/full_data_monthly_v2018_05.nc.gz')
df_prep = ds['precip'].mean(dim = ['lat','lon']).to_dataframe().reset_index()
df_prep.rename(columns={'time':'Date','precip':'Precip_in_mm'},inplace=True)
df_prep['Date'] = pd.to_datetime(df_prep['Date'])
precip = df_prep[df_prep['Date'] >= '2014-01-01'].reset_index(drop=True)
print(precip.shape)
print(precip)

ds = pd.read_csv('./DATASETS/NASA_Global_Temp_monthly.csv')
ds = ds[ds['Source']=='GISTEMP'] # Row filtering to include only rows with source as GISTEMP
ds['Year'] = pd.to_datetime(ds['Year'])
df_temp = ds[['Year','Mean']].rename(columns={'Mean':'Temp_Anomaly_°C','Year':'Date'}).reset_index(drop =True)
temp = df_temp[(df_temp['Date'] >= '2014-01-01' ) & (df_temp['Date'] <= '2016-12-01')].reset_index(drop=True)
print(temp.shape)
print(temp)

ds = pd.read_csv('./DATASETS/co2_monthly_mm_mlo.csv',comment = '#',header = 0)
ds.columns = ['Year','Month','Decimal Date','Average','Deseasonalized','NDays','StDev','Uncertainty']
ds['Date'] = pd.to_datetime(dict(year=ds.Year, month=ds.Month, day=1)) # Create a date column from year and month and set the day to 1
df_co2 = ds[['Date', 'Deseasonalized']].rename(columns={'Deseasonalized': 'CO2_in_ppm'})
co2 = df_co2[(df_co2['Date'] >= '2014-01-01' ) & (df_co2['Date'] <= '2016-12-01')].reset_index(drop=True)
print(co2.shape)
print(co2)

date_range = pd.date_range(start='2014-01-01', end='2016-12-01', freq='MS')
vol = [format(0.0, '.4f') for i in range(36)]
volc = pd.DataFrame({'Date': date_range, 'Volcanic_Global': vol})
print(volc)

df_nino = pd.read_csv('./DATASETS/nino34.long.anom.csv', skiprows=1)
df_nino.columns = ['Date', 'Nino34'] + [f'extra_{i}' for i in range(len(df_nino.columns)-2)]
df_nino['Date'] = df_nino['Date'].astype(str).str.rstrip(',')
df_nino['Date'] = pd.to_datetime(df_nino['Date'])
df_nino_long = df_nino[['Date', 'Nino34']].copy()
df_nino_long['Nino34'] = pd.to_numeric(df_nino_long['Nino34'], errors='coerce')
df_nino_long['Nino34'] = df_nino_long['Nino34'].replace(-99.99, np.nan)
nino = df_nino_long[(df_nino_long['Date'] >= '2014-01-01' ) & (df_nino_long['Date'] <= '2016-12-01')].reset_index(drop=True).rename(columns={'Nino34': 'Nino34_in_°C'})
print(nino)

tsi_files = glob.glob("./DATASETS/tsi_*.nc")  
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
print(combined_tsi)

d1 = pd.merge(temp,co2, on='Date', how='inner')
d2 = pd.merge(d1, precip, on='Date', how='inner')
d3 = pd.merge(d2, nino, on='Date', how='inner')
d4 = pd.merge(d3, volc, on='Date', how='inner')
final_test = pd.merge(d4, combined_tsi, on='Date', how='inner')
print(final_test)
final_test.to_csv('Main_Test.csv', index=False)