# -*- coding: utf-8 -*-
"""
Created on Sat May 24 16:48:01 2025

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

from collections import defaultdict

n_values = [12, 24, 36, 48]
wind_dir_counts_raw = {n: defaultdict(int) for n in n_values}
wind_dir_counts_active = {n: defaultdict(int) for n in n_values}
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

n = 12
ang_interval = 360/n
# Define custom bin edges centered around the desired intervals
bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
bin_labels = np.arange(0.0, 360.0, ang_interval)

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
    df_energy_wind_dir_raw = df_energy_raw.copy()
    for n in n_values:
        ang_interval = 360 / n
        bin_edges = np.arange(-ang_interval / 2, 360 + ang_interval / 2, ang_interval)
        bin_labels = np.arange(0, 360, ang_interval)
        
        df_energy_wind_dir_raw['dir_bin'] = pd.cut(
            df_energy_wind_dir_raw['wind_dir'],
            bins=bin_edges,
            labels=bin_labels,
            right=False,
            include_lowest=True
        )
    
        bin_counts = df_energy_wind_dir_raw['dir_bin'].value_counts()
        for direction, count in bin_counts.items():
            wind_dir_counts_raw[n][direction] += count
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

#%% Count
df_wind_dir_filt=df_energy_filtered2_all.copy()
for n in n_values:
    ang_interval = 360 / n
    bin_edges = np.arange(-ang_interval / 2, 360 + ang_interval / 2, ang_interval)
    bin_labels = np.arange(0, 360, ang_interval)
# Filtered counts
    df_wind_dir_filt['dir_bin'] = pd.cut(
        df_wind_dir_filt['wind_dir'],
        bins=bin_edges,
        labels=bin_labels,
        right=False,
        include_lowest=True
    )
    for direction, count in df_wind_dir_filt['dir_bin'].value_counts().items():
        wind_dir_counts_active[n][direction] += count
#%% Save
for n in n_values:
    bin_labels = sorted(wind_dir_counts_raw[n].keys())  # Ensure ordered bins
    df_counts = pd.DataFrame({
        'direction_bin': bin_labels,
        'initial_len': [wind_dir_counts_raw[n][b] for b in bin_labels],
        'active_len': [wind_dir_counts_active[n][b] for b in bin_labels],
    })

    save_path = os.path.join(save_folder, f"{n}_wind_direction_counts.csv")
    df_counts.to_csv(save_path, index=False)
    print(f"Saved wind direction counts for {n} bins to {save_path}")