# -*- coding: utf-8 -*-
"""
Created on Sun May 11 15:39:51 2025

@author: erica
"""

import pandas as pd
import numpy as np
import glob
import os
from datetime import datetime
import matplotlib.pyplot as plt

#%% Import and save
# Function to extract turbine ID and year from filename
def extract_turbine_year(filename):
    parts = os.path.basename(filename).split("_")
    turbine_id = int(parts[3])  # Extract turbine number
    year = parts[4][:4]  # Extract year
    return turbine_id, year

n_wt=14
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

# Function to detect NaN sequences and extract neighboring values
def analyze_nan_sequences(df, turbine_id, year):
    results = []
    # Identify consecutive NaNs
    is_na = df['wind_speed'].isna()
    na_groups = (is_na != is_na.shift()).cumsum()
    na_sequences = is_na.groupby(na_groups).agg(['size', 'first'])
    
    # Filter sequences with >6 NaNs
    long_nans = na_sequences[(na_sequences['size'] > 6) & (na_sequences['first'])]
    
    for idx, seq in long_nans.iterrows():
        # Get start/end positions of the NaN sequence
        nan_group = na_groups[na_groups == idx]
        nan_start = nan_group.index[0]
        nan_end = nan_group.index[-1]
        
        # Get last valid before gap
        before_gap = df.loc[:nan_start - 1, 'wind_speed'].last_valid_index()
        ws_before = df.loc[before_gap, 'wind_speed'] if before_gap is not None else np.nan
        
        # Get first valid after gap
        after_gap = df.loc[nan_end + 1:, 'wind_speed'].first_valid_index()
        ws_after = df.loc[after_gap, 'wind_speed'] if after_gap is not None else np.nan
        
        # Calculate stats
        if not np.isnan(ws_before) and not np.isnan(ws_after):
            ws_mean = np.mean([ws_before, ws_after])
            ws_diff = abs(ws_after - ws_before)
            is_relevant = (abs(ws_mean - ws_before) < 2) and (abs(ws_mean - ws_after) < 2)
        else:
            ws_mean = ws_diff = np.nan
            is_relevant = False
            
        results.append({
            'turbine_id': turbine_id,
            'year': year,
            'nan_sequence_length': seq['size'],
            'gap_start': df.loc[nan_start, 'date_time'],
            'gap_end': df.loc[nan_end, 'date_time'],
            'ws_before': ws_before,
            'ws_after': ws_after,
            'ws_mean': ws_mean,
            'ws_diff': ws_diff,
            'is_mean_relevant': is_relevant
        })
    
    return results

# Main processing loop
nan_analysis_records = []

for file in all_files:
    turbine_id, year = extract_turbine_year(file)
    if turbine_id >= 3:
        turbine_id = turbine_id - 1
    
    df = pd.read_csv(file, header=0, usecols=["# Date and time", "Wind speed (m/s)", "Wind direction (°)", "Energy Export (kWh)", "Power (kW)"], skiprows=9, parse_dates=["# Date and time"])
    df.columns = ['date_time', 'wind_speed', 'wind_dir', 'energy', 'power']
    
    df["turbine_id"] = turbine_id
    df["year"] = int(year)
    
    # Analyze NaN sequences
    nan_results = analyze_nan_sequences(df, turbine_id, year)
    nan_analysis_records.extend(nan_results)

# Save results to CSV
nan_analysis_df = pd.DataFrame(nan_analysis_records)
nan_analysis_df.to_csv(f'{save_folder}/wind_speed_nan_analysis.csv', index=False)
print(f"Saved NaN sequence analysis to {save_folder}/wind_speed_nan_analysis.csv")