# -*- coding: utf-8 -*-
"""
Created on Tue May 13 09:11:15 2025

@author: erica
"""

# import and save SCADA data

import pandas as pd
import numpy as np
from scipy.stats import zscore  # For outlier removal
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt

#%%
u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050]
    
#%% Import and save
# Function to extract turbine ID and year from filename
def extract_turbine_year(filename):
    parts = os.path.basename(filename).split("_")
    turbine_id = int(parts[3])  # Extract turbine number
    year = parts[4][:4]  # Extract year
    return turbine_id, year

n_wt = 14
cut_in_ws = 3.5
cut_out_ws = 25
nominal_ws = 14.5
rated_power = 2050
# Get current date and time
now = datetime.now()
date_folder = now.strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
time_str = now.strftime("%H_%M")  # Format: HH_MM

# Create folder path (in current directory)
save_folder = os.path.join('D:/Thesis/Data/Penmanshiel/', date_folder)

# Create folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Load all turbine data
parent_folder = "D:/Thesis/Data/Penmanshiel/"
all_files = glob.glob(os.path.join(parent_folder, "*", "Turbine_Data_Penmanshiel_*.csv"))
df_windspeed_list = []
df_energy_raw_list = []
df_energy_error1_list = []
df_energy_pitch_list = []
metadata_records = []
energy_adj = []
initial_len = 0
valid_len = 0
for file in all_files:
    turbine_id, year = extract_turbine_year(file)
    if turbine_id >= 3:
        turbine_id = turbine_id - 1
    if int(year) >=2019:
        df = pd.read_csv(file, header=0, usecols=["# Date and time", "Wind speed (m/s)", "Wind direction (°)", "Energy Export (kWh)", "Power (kW)","Blade angle (pitch position) A (°)"], skiprows=9, parse_dates=["# Date and time"])
        df.columns = ['date_time', 'wind_speed', 'wind_dir', 'energy', 'power', 'pitch']
    else:
        df = pd.read_csv(file, header=0, usecols=["# Date and time", "Wind speed (m/s)", "Wind direction (°)", "Energy Export (kWh)", "Power (kW)"], skiprows=9, parse_dates=["# Date and time"])
        df.columns = ['date_time', 'wind_speed', 'wind_dir', 'energy', 'power']
    df["turbine_id"] = turbine_id
    df["year"] = int(year)
    df['date_time'] = pd.to_datetime(df['date_time'])
    
    df_energy = df.copy()
    df_energy_raw = df_energy.dropna()
    df_energy_error1 = df_energy_raw[(df_energy_raw['wind_speed'] < cut_in_ws) | ((df_energy_raw['power'] > 2.7) & (df_energy_raw['wind_speed'] >= cut_in_ws))]
    if int(year)>=2019:
        df_energy_pitch = df_energy_error1[((df_energy_error1['pitch'] < 40)& (df_energy_error1['wind_speed']>=nominal_ws)&(df_energy_error1['power'] > 0.9*rated_power)) | 
                              ((df_energy_error1['pitch'] < 7)&(df_energy_error1['wind_speed'] <nominal_ws))|
                              (df_energy_error1['wind_speed'] <cut_in_ws)]
        df_energy_pitch = df_energy_pitch.drop(columns=['pitch'])
    else:
        df_energy_pitch=df_energy_error1
        
    df_energy_raw_list.append(df_energy_raw)
    df_energy_error1_list.append(df_energy_error1)
    df_energy_pitch_list.append(df_energy_pitch)


#%%
df_energy_raw_all = pd.concat(df_energy_raw_list, ignore_index=True)
df_energy_raw_all.set_index("date_time", inplace=True)
df_energy_raw_all.sort_index(inplace=True)

df_energy_error1_all = pd.concat(df_energy_error1_list, ignore_index=True)
df_energy_error1_all.set_index("date_time", inplace=True)
df_energy_error1_all.sort_index(inplace=True)

df_energy_pitch_all = pd.concat(df_energy_pitch_list, ignore_index=True)
df_energy_pitch_all.set_index("date_time", inplace=True)
df_energy_pitch_all.sort_index(inplace=True)

#%% ALL FILTERING
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=[10,3])

# df_energy_raw_all = df_energy_raw_all[df_energy_raw_all['year']==2020]
# df_energy_error1_all = df_energy_error1_all[df_energy_error1_all['year']==2020]
# df_energy_pitch_all = df_energy_pitch_all[df_energy_pitch_all['year']==2020]

plt.plot(df_energy_raw_all['wind_speed'],df_energy_raw_all['power'],'.',label='Raw SCADA data',markersize=1)

plt.plot(df_energy_error1_all['wind_speed'],df_energy_error1_all['power'],'.',label='Zero-filtered SCADA',markersize=1)

plt.plot(df_energy_pitch_all['wind_speed'],df_energy_pitch_all['power'],'.',label='Pitch Filtering',markersize=1)

# plt.xlabel('Wind speed (m/s)')
# plt.ylabel('Power (W)')
# plt.legend(loc='upper left')
# plt.show()
#plt.xlabel('Wind speed (m/s)')
#plt.ylabel('Power (W)')


power_bin_size=1
df_energy_pitch_all['power_bin'] = np.floor(df_energy_pitch_all['power'] / power_bin_size).astype(int) * power_bin_size
# Process each bin separately to minimize memory usage
cleaned_chunks = []

for bin_val, group in df_energy_pitch_all.groupby('power_bin'):
    if bin_val < 0.85 * rated_power:
        z_scores = zscore(group['wind_speed'])
        mask = np.abs(z_scores) <= 3
        cleaned_chunks.append(group[mask])
    else:
        cleaned_chunks.append(group) 
    
# Combine all cleaned chunks
df_energy_filtered1_all = pd.concat(cleaned_chunks)
df_energy_filtered1_all = df_energy_filtered1_all.drop(columns=['power_bin'])

plt.plot(df_energy_filtered1_all['wind_speed'],df_energy_filtered1_all['power'],'.',label='LOF Power',markersize=1)




wind_bin_size=0.2
df_energy_filtered1_all['wind_bin'] = np.floor(df_energy_filtered1_all['wind_speed'] / wind_bin_size).astype(int) * wind_bin_size
# Process each bin separately to minimize memory usage
cleaned_chunks = []

for bin_val, group in df_energy_filtered1_all.groupby('wind_bin'):
    if bin_val>nominal_ws:
        z_scores = zscore(group['power'])
        # Keep only rows within 3 standard deviations
        mask = np.abs(z_scores) <= 5
        cleaned_chunks.append(group[mask])
    else:
        cleaned_chunks.append(group)
    
# Combine all cleaned chunks
df_energy_filtered2_all = pd.concat(cleaned_chunks)
df_energy_filtered2_all = df_energy_filtered2_all.drop(columns=['wind_bin'])

plt.plot(df_energy_filtered2_all['wind_speed'],df_energy_filtered2_all['power'],'.',label='LOF Wind Speed',markersize=1)

plt.plot(u,power,label='Theoretical Power curve')

plt.vlines(nominal_ws,0,2100,linestyle='--',colors='grey')
plt.hlines(0.85 * rated_power,0,26,linestyle='--',colors='grey')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Power (W)')
leg=plt.legend(loc='center right', markerscale=10)
plt.show()


#%% Raw data
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=[10,3])

plt.plot(df_energy_raw_all['wind_speed'],df_energy_raw_all['power'],'.',label='Raw SCADA data',markersize=1)
plt.plot(u,power,label='Theoretical Power Curve')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Power (W)')
plt.legend(loc='upper left',markerscale=10)
plt.show()

#%% Error1 removal
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=[10,3])

plt.plot(df_energy_raw_all['wind_speed'],df_energy_raw_all['power'],'.',label='Raw SCADA data',markersize=1)

plt.plot(df_energy_error1_all['wind_speed'],df_energy_error1_all['power'],'.',label='Zero-filtered SCADA',markersize=1)

plt.xlabel('Wind speed (m/s)')
plt.ylabel('Power (W)')
plt.legend(loc='upper left',markerscale=10)
plt.show()

#%% Pitch filtering only 2019 and 2020
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=[10,3])

df_energy_raw_all_y = df_energy_raw_all[(df_energy_raw_all['year']==2020) | (df_energy_raw_all['year']==2019)]
df_energy_error1_all_y = df_energy_error1_all[(df_energy_error1_all['year']==2020)| (df_energy_error1_all['year']==2019)]
df_energy_pitch_all_y = df_energy_pitch_all[(df_energy_pitch_all['year']==2020)| (df_energy_pitch_all['year']==2019)]

#plt.plot(df_energy_raw_all_y['wind_speed'],df_energy_raw_all_y['power'],'.',label='Raw SCADA data',markersize=1)

plt.plot(df_energy_error1_all_y['wind_speed'],df_energy_error1_all_y['power'],'.',label='Zero-filtered SCADA',markersize=1)

plt.plot(df_energy_pitch_all_y['wind_speed'],df_energy_pitch_all_y['power'],'.',label='Pitch Filtering',markersize=1)

plt.xlabel('Wind speed (m/s)')
plt.ylabel('Power (W)')
plt.legend(loc='upper left',markerscale=10)
plt.show()

#%% Pitch filtering with everything
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=[10,3])

#plt.plot(df_energy_raw_all['wind_speed'],df_energy_raw_all['power'],'.',label='Raw SCADA data',markersize=1)

plt.plot(df_energy_error1_all['wind_speed'],df_energy_error1_all['power'],'.',label='Zero-filtered SCADA',markersize=1)

plt.plot(df_energy_pitch_all['wind_speed'],df_energy_pitch_all['power'],'.',label='Pitch Filtering',markersize=1)

plt.xlabel('Wind speed (m/s)')
plt.ylabel('Power (W)')
plt.legend(loc='upper left',markerscale=10)
plt.show()
#plt.xlabel('Wind speed (m/s)')
#plt.ylabel('Power (W)')

#%% LOF Power
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=[10,3])
plt.plot(df_energy_pitch_all['wind_speed'],df_energy_pitch_all['power'],'.',label='Pitch Filtering',markersize=1)
power_bin_size=1
df_energy_pitch_all['power_bin'] = np.floor(df_energy_pitch_all['power'] / power_bin_size).astype(int) * power_bin_size
# Process each bin separately to minimize memory usage
cleaned_chunks = []

for bin_val, group in df_energy_pitch_all.groupby('power_bin'):
    if bin_val < 0.85 * rated_power:
        z_scores = zscore(group['wind_speed'])
        mask = np.abs(z_scores) <= 3
        cleaned_chunks.append(group[mask])
    else:
        cleaned_chunks.append(group) 
    
# Combine all cleaned chunks
df_energy_filtered1_all = pd.concat(cleaned_chunks)
df_energy_filtered1_all = df_energy_filtered1_all.drop(columns=['power_bin'])

plt.plot(df_energy_filtered1_all['wind_speed'],df_energy_filtered1_all['power'],'.',label='LOF Power',markersize=1)
#plt.plot(u,power,label='Theoretical Power curve')
plt.hlines(0.85 * rated_power,0,26,linestyle='--',colors='grey')
#plt.hlines(410,0,26,linestyle='--',colors='magenta')
#plt.hlines(410+power_bin_size,0,26,linestyle='--',colors='magenta')
#plt.ylim((400,420))
#plt.xlim((5,10))
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Power (W)')
plt.legend(loc='upper right',markerscale=10)
plt.show()

#%% LOF Wind speed
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=[10,3])
plt.plot(df_energy_filtered1_all['wind_speed'],df_energy_filtered1_all['power'],'.',label='LOF Power',markersize=1)
wind_bin_size=0.2
df_energy_filtered1_all['wind_bin'] = np.floor(df_energy_filtered1_all['wind_speed'] / wind_bin_size).astype(int) * wind_bin_size
# Process each bin separately to minimize memory usage
cleaned_chunks = []

for bin_val, group in df_energy_filtered1_all.groupby('wind_bin'):
    if bin_val>nominal_ws:
        z_scores = zscore(group['power'])
        # Keep only rows within 3 standard deviations
        mask = np.abs(z_scores) <= 5
        cleaned_chunks.append(group[mask])
    else:
        cleaned_chunks.append(group)
    
# Combine all cleaned chunks
df_energy_filtered2_all = pd.concat(cleaned_chunks)
df_energy_filtered2_all = df_energy_filtered2_all.drop(columns=['wind_bin'])

plt.plot(df_energy_filtered2_all['wind_speed'],df_energy_filtered2_all['power'],'.',label='LOF Wind Speed',markersize=1)

plt.vlines(nominal_ws,0,2100,linestyle='--',colors='grey')
#plt.vlines(17,0,2100,linestyle='--',colors='magenta')
#plt.vlines(17+wind_bin_size,0,2100,linestyle='--',colors='magenta')
#plt.hlines(0.85 * rated_power,0,26,linestyle='--',colors='grey')
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Power (W)')
leg=plt.legend(loc='center right', markerscale=10)
plt.show()

#%% Power
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=[7,5])

plt.plot(df_energy_raw_all['wind_speed'],df_energy_raw_all['power'],'.',label='Raw SCADA data',markersize=1)
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Power (W)')
#plt.legend(loc='upper left',markerscale=10)
plt.show()

#%% Energy
plt.rcParams.update({'font.size': 12})
plt.figure(figsize=[7,5])

plt.plot(df_energy_raw_all['wind_speed'],df_energy_raw_all['energy'],'.',label='Raw SCADA data',markersize=1)
plt.hlines(310,0,30,linestyles='--',color='grey')
plt.hlines(412,0,30,linestyles='--',color='grey')
plt.yticks([0,100,200,310,412,500],['0','100','200','310','412','500'])
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Energy Export (Wh)')
#plt.legend(loc='upper left',markerscale=10)
plt.show()