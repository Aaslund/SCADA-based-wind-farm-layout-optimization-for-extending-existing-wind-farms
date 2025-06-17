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
from collections import defaultdict
from scipy.interpolate import interp1d

n_values = [12, 24, 36]
wind_dir_counts_raw = {n: defaultdict(int) for n in n_values}
wind_dir_counts_active = {n: defaultdict(int) for n in n_values}
#%%
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    elif month in [9, 10, 11]:
        return 'Autumn'
    
#%% Import and save
# Function to extract turbine ID and year from filename
def extract_turbine_year(filename):
    parts = os.path.basename(filename).split("_")
    turbine_id = int(parts[3])  # Extract turbine number
    year = parts[4][:4]  # Extract year
    return turbine_id, year

def circular_mean(degrees):
    radians = np.deg2rad(degrees)
    sin_sum = np.mean(np.sin(radians))
    cos_sum = np.mean(np.cos(radians))
    mean_angle = np.arctan2(sin_sum, cos_sum)
    return np.rad2deg(mean_angle) % 360

from collections import defaultdict

# Dictionary to hold removed rows per (turbine_id, year, season)
removed_bin_energy = defaultdict(lambda: {'actual': 0.0, 'expected': 0.0})

def accumulate_removed_energy(df_removed):
    # Group removed rows by turbine_id, year, and season
    for (turbine_id, year, season), group in df_removed.groupby(['turbine_id', df_removed.index.year, 'season']):
        actual_energy = (group['power'] * (10/60)).sum()
        expected_energy = (interp_curve(group['wind_speed']) * (10/60)).sum()
        removed_bin_energy[(turbine_id, year, season)]['actual'] += actual_energy
        removed_bin_energy[(turbine_id, year, season)]['expected'] += expected_energy

n_wt = 14
cut_in_ws = 3.5
cut_out_ws = 25
nominal_ws = 14.5
rated_power = 2050
u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25.1, 30]
power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040,
         2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 0, 0]
interp_curve = interp1d(u, power, bounds_error=False, fill_value=0)
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
df_energy_list = []
metadata_records = []
energy_adj = []
initial_len = 0
valid_len = 0
season_list=['Winter', 'Spring', 'Summer', 'Autumn']
for file in all_files:
    turbine_id, year = extract_turbine_year(file)
    if turbine_id >= 3:
        turbine_id = turbine_id - 1
    if int(year) >=2019:
        df = pd.read_csv(file, header=0, usecols=["# Date and time", "Wind speed (m/s)", "Wind speed, Standard deviation (m/s)","Wind direction (°)", "Energy Export (kWh)", "Power (kW)","Blade angle (pitch position) A (°)"], skiprows=9, parse_dates=["# Date and time"])
        df.columns = ['date_time', 'wind_speed', 'wind_speed_std', 'wind_dir', 'energy', 'power', 'pitch']
    else:
        df = pd.read_csv(file, header=0, usecols=["# Date and time", "Wind speed (m/s)", "Wind speed, Standard deviation (m/s)","Wind direction (°)", "Energy Export (kWh)", "Power (kW)"], skiprows=9, parse_dates=["# Date and time"])
        df.columns = ['date_time', 'wind_speed', 'wind_speed_std','wind_dir', 'energy', 'power']
    df["turbine_id"] = turbine_id
    df["year"] = int(year)
    df['date_time'] = pd.to_datetime(df['date_time'])
    df['season'] = df['date_time'].dt.month.apply(get_season)
    for season_i in season_list:
        df_season = df[df['season'] == season_i].copy()  # Safe copy
        nan_count_10min_power = df_season['power'].isna().sum()
        nan_count_10min_energy = df_season['energy'].isna().sum()
        nan_count_10min_ws = df_season['wind_speed'].isna().sum()
        
        total_rows = len(df)     
        
        df_season.loc[:, 'TI'] = df_season['wind_speed_std'] / df_season['wind_speed']
        df_windspeed = df_season[['date_time', 'wind_speed', 'TI','wind_dir', 'turbine_id', 'year', 'season']].copy()
        df_windspeed = df_windspeed.dropna()
        df_windspeed_hourly_init = df_windspeed.resample('h', on='date_time').agg({
            'wind_speed': 'mean',
            'TI':'mean',
            'wind_dir':circular_mean,
            'turbine_id': 'first',
            'year': 'first',
            'season': 'first'
            }).reset_index()
        df_windspeed_hourly_all = df_windspeed_hourly_init.copy()
        df_windspeed_hourly_all = df_windspeed_hourly_all.dropna() # NaNs in the months between January and November for the winter case
        total_rows_hourly = len(df_windspeed_hourly_all)
        df_windspeed_hourly = df_windspeed_hourly_all[(df_windspeed_hourly_all['wind_speed'] > 0.5) & (df_windspeed_hourly_all['wind_speed'] < 30)]
        windspeed_rows_after_filtering = len(df_windspeed_hourly)
        
        df_energy = df_season.copy()
        initial_len=len(df_energy)
        df_energy = df_energy.dropna()
        valid_len=len(df_energy)
        df_energy = df_energy[(df_energy['wind_speed'] > 0.5) & (df_energy['wind_speed'] < 30)]
        for n in n_values:
            ang_interval = 360 / n
            bin_edges = np.arange(-ang_interval / 2, 360 + ang_interval / 2, ang_interval)
            bin_labels = np.arange(0, 360, ang_interval)
            df_energy_wind_dir_raw = df_energy.copy()
            df_energy_wind_dir_raw.loc[:,'wind_dir'] = df_energy_wind_dir_raw['wind_dir'].apply(lambda wd: wd - 360 if wd >= 360-ang_interval/2 else wd)
            df_energy_wind_dir_raw['dir_bin'] = pd.cut(
                df_energy_wind_dir_raw['wind_dir'],
                bins=bin_edges,
                labels=bin_labels,
                right=False,
                include_lowest=True
            ).astype(float)  # Optional, for compatibility
        
            # Count per direction, season, year
            grouped = df_energy_wind_dir_raw.groupby(['dir_bin', 'season', 'year'])
            for (direction, season, year), group in grouped:
                wind_dir_counts_raw[n][(direction, season, year)] += len(group)

        df_energy = df_energy[(df_energy['wind_speed'] < cut_in_ws)| ((df_energy['power'] >= 2.7) & (df_energy['wind_speed'] >= cut_in_ws))]
        df_energy['power'] = df_energy['power'].clip(lower=0, upper=rated_power*1.02)
        removed_energy = 0
        expected_energy = 0
        if int(year)>=2019:
            cond_above_nominal = (df_energy['pitch'] < 40) & \
                     (df_energy['wind_speed'] >= nominal_ws) & \
                     (df_energy['power'] > 0.9 * rated_power)

            cond_below_nominal = (df_energy['pitch'] < 7) & \
                                 (df_energy['wind_speed'] < nominal_ws)
            
            cond_cut_in = df_energy['wind_speed'] < cut_in_ws
            
            # Combine conditions into a single valid mask
            valid_mask = cond_above_nominal | cond_below_nominal | cond_cut_in
            
            # Store removed entries and filter the DataFrame
            removed = df_energy[~valid_mask]
            df_energy = df_energy[valid_mask].drop(columns=['pitch'])
            removed_energy = (removed['power'] * (10/60)).sum()
            expected_power = interp_curve(removed['wind_speed'])
            expected_energy = (expected_power * (10/60)).sum()
        metadata_records.append({
           'turbine_id': turbine_id,
           'year': year,
           'season': season_i,
           'nan_count_10min_ws': nan_count_10min_ws,
           'windspeed_frac_nan': (total_rows-nan_count_10min_ws)/total_rows,
           'windspeed_frac_outlier': windspeed_rows_after_filtering/total_rows_hourly,
           'nan_count_10min_power': nan_count_10min_power,
           'nan_count_10min_energy': nan_count_10min_energy,
           'initial_len': initial_len,
           'valid_len': valid_len,
           'active_len': 'NaN',
           'removed_energy_actual_pitch': removed_energy,
           'removed_energy_expected_pitch': expected_energy
        })  
            
        df_windspeed_list.append(df_windspeed_hourly)
        df_energy_list.append(df_energy)

#%%
# Combine all data into a single DataFrame
df_windspeed_all = pd.concat(df_windspeed_list, ignore_index=True)
df_windspeed_all.set_index("date_time", inplace=True)
df_windspeed_all.sort_index(inplace=True)

#%%
df_energy_all = pd.concat(df_energy_list, ignore_index=True)
df_energy_all.set_index("date_time", inplace=True)
df_energy_all.sort_index(inplace=True)

#%% LOF
# Power binning
power_bin_size=1
df_energy_all['power_bin'] = np.floor(df_energy_all['power'] / power_bin_size).astype(int) * power_bin_size
# Process each bin separately to minimize memory usage
cleaned_chunks = []
removed_rows = []
for bin_val, group in df_energy_all.groupby('power_bin'):
    if bin_val < 0.85 * rated_power:
        z_scores = zscore(group['wind_speed'])
        mask = np.abs(z_scores) <= 3
        removed = group[~mask]
        removed_rows.append(removed)
        accumulate_removed_energy(removed)
        cleaned_chunks.append(group[mask])
    else:
        cleaned_chunks.append(group) 
    
# Combine all cleaned chunks
df_energy_all = pd.concat(cleaned_chunks)
df_energy_all = df_energy_all.drop(columns=['power_bin'])

# Wind speed binning

wind_bin_size=0.2
df_energy_all['wind_bin'] = np.floor(df_energy_all['wind_speed'] / wind_bin_size).astype(int) * wind_bin_size
# Process each bin separately to minimize memory usage
cleaned_chunks = []

for bin_val, group in df_energy_all.groupby('wind_bin'):
    if bin_val>nominal_ws:
        z_scores = zscore(group['power'])
        # Keep only rows within 5 standard deviations
        mask = np.abs(z_scores) <= 5
        removed = group[~mask]
        removed_rows.append(removed)
        accumulate_removed_energy(removed)
        cleaned_chunks.append(group[mask])
    else:
        cleaned_chunks.append(group)
    
# Combine all cleaned chunks
df_energy_all = pd.concat(cleaned_chunks)
df_energy_all = df_energy_all.drop(columns=['wind_bin'])

removed_df = pd.concat(removed_rows)
interval_h = 10 / 60  # 10-minute intervals
removed_energy = (removed_df['power'] * interval_h).sum()
expected_energy = (interp_curve(removed_df['wind_speed']) * interval_h).sum()


#%% Wind dir counting
for n in n_values:
    ang_interval = 360 / n
    bin_edges = np.arange(-ang_interval / 2, 360 + ang_interval / 2, ang_interval)
    bin_labels = np.arange(0, 360, ang_interval)
    df_wind_dir_filt = df_energy_all.copy()
    df_wind_dir_filt.loc[:,'wind_dir'] = df_wind_dir_filt['wind_dir'].apply(lambda wd: wd - 360 if wd >= 360-ang_interval/2 else wd)
    df_wind_dir_filt['dir_bin'] = pd.cut(
        df_wind_dir_filt['wind_dir'],
        bins=bin_edges,
        labels=bin_labels,
        right=False,
        include_lowest=True
    ).astype(float)  # Optional

    grouped = df_wind_dir_filt.groupby(['dir_bin', 'season', 'year'])
    for (direction, season, year), group in grouped:
        wind_dir_counts_active[n][(direction, season, year)] += len(group)
#%%
# Saving
df_windspeed_all=df_windspeed_all.reset_index()
for year, group in df_windspeed_all.groupby(df_windspeed_all['date_time'].dt.year):
    # Create filename for this year
    filename = f"df_windspeed_all_{year}.csv"
    
    # Full path to save
    save_path = os.path.join(save_folder, filename)
    
    if os.path.exists(save_path):  # Replace with your file path
        os.remove(save_path)  # Deletes the file
    # Save the DataFrame for this year
    group.to_csv(save_path, index=False)
    print(f"Saved data for year {year} to {save_path}")

filename = "df_windspeed_tot.csv"

# Full path to save
save_path = os.path.join(save_folder, filename)

if os.path.exists(save_path):  # Replace with your file path
    os.remove(save_path)  # Deletes the file
# Save the DataFrame for this year
df_windspeed_all.to_csv(save_path, index=False)
print("Saved data for all years")
#%%
                    
df_energy_all=df_energy_all.reset_index()
for year, groupYear in df_energy_all.groupby(df_energy_all['date_time'].dt.year):
    # Create filename for this year
    filename = f"df_energy_all_{year}.csv"
    
    # Full path to save
    save_path = os.path.join(save_folder, filename)
    
    if os.path.exists(save_path):  # Replace with your file path
        os.remove(save_path)  # Deletes the file
    # Save the DataFrame for this year
    groupYear.to_csv(save_path, index=False)
    print(f"Saved data for year {year} to {save_path}")
    for turbine_id,groupTurbine in groupYear.groupby(groupYear['turbine_id']):
        for season_group, groupSeason in groupTurbine.groupby(groupTurbine['season']):
            for record in metadata_records:
                if (
                    record['turbine_id'] == turbine_id
                    and record['year'] == year
                    and record['season'] == season_group
                ):
                    record['active_len'] = len(groupSeason)
                    key = (turbine_id, year, season_group)
                    if key in removed_bin_energy:
                        record['removed_energy_actual_bin'] = removed_bin_energy[key]['actual']
                        record['removed_energy_expected_bin'] = removed_bin_energy[key]['expected']
                    else:
                        record['removed_energy_actual_bin'] = 0.0
                        record['removed_energy_expected_bin'] = 0.0
                    break

metadata_df = pd.DataFrame(metadata_records)
metadata_df.to_csv(f'{save_folder}/turbine_year_metadata.csv', index=False)

#%% Save wind direction count
for n in n_values:
    all_keys = set(wind_dir_counts_raw[n].keys()).union(wind_dir_counts_active[n].keys())
    data = []
    for key in sorted(all_keys):
        direction, season, year = key
        initial_len = wind_dir_counts_raw[n].get(key, 0)
        active_len = wind_dir_counts_active[n].get(key, 0)
        data.append({
            'direction_bin': direction,
            'season': season,
            'year': year,
            'initial_len': initial_len,
            'active_len': active_len
        })
    df_combined = pd.DataFrame(data)
    save_path = os.path.join(save_folder, f"{n}_wind_direction_seasonal_counts.csv")
    df_combined.to_csv(save_path, index=False)
    print(f"Saved combined wind direction seasonal counts for {n} bins to {save_path}")

#%%
filename = "df_energy_tot.csv"

# Full path to save
save_path = os.path.join(save_folder, filename)

if os.path.exists(save_path):  # Replace with your file path
    os.remove(save_path)  # Deletes the file
# Save the DataFrame for this year
df_energy_all.to_csv(save_path, index=False)
print("Saved data for all years")