# -*- coding: utf-8 -*-
"""
Created on Mon Feb  3 11:11:08 2025

@author: erica
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore  # For outlier removal
import glob
import os

# Function to extract turbine ID and year from filename
def extract_turbine_year(filename):
    parts = os.path.basename(filename).split("_")
    turbine_id = int(parts[3])  # Extract turbine number
    year = parts[4][:4]  # Extract year
    return turbine_id, year

# Load all turbine data
parent_folder = "D:/Thesis/Data/Penmanshiel/"
all_files = glob.glob(os.path.join(parent_folder, "*", "Turbine_Data_Penmanshiel_*.csv"))
df_list = []
for file in all_files:
    turbine_id, year = extract_turbine_year(file)
    df = pd.read_csv(file, header=0, usecols=["# Date and time", "Wind speed (m/s)", "Wind direction (°)", "Power (kW)"], skiprows=9, parse_dates=["# Date and time"])
    df.columns = ['date_time', 'wind_speed', 'wind_dir', 'power']
    df["turbine_id"] = turbine_id if turbine_id < 3 else turbine_id - 1
    df["year"] = int(year)
    df_list.append(df)

# Combine all data into a single DataFrame
df_all = pd.concat(df_list, ignore_index=True)
df_all.set_index("date_time", inplace=True)
df_all.sort_index(inplace=True)

# === 1. Outlier Removal ===
df_all = df_all[(df_all['wind_speed'] > 0.5) & (df_all['wind_speed'] < 30)]
df_all = df_all[np.abs(zscore(df_all['wind_speed'])) < 3]  # Remove extreme outliers

# === 2. Time Averaging ===
df_hourly = df_all.groupby(['turbine_id', pd.Grouper(freq='h')]).mean().reset_index()

# Adjust wind direction so that 345°–360° is mapped to -15°–0°
df_hourly['wind_dir_adjusted'] = df_hourly['wind_dir'].apply(lambda wd: wd - 360 if wd >= 345 else wd)

# === Plotting ===
# Wind Speed vs. Time
plt.figure(figsize=(12, 6))
for turbine_id in df_hourly['turbine_id'].unique():
    turbine_data = df_hourly[df_hourly['turbine_id'] == turbine_id]
    plt.plot(turbine_data['date_time'], turbine_data['wind_speed'], label=f"Wind turbine {turbine_id}")

plt.xlabel("Time")
plt.ylabel("Wind speed (m/s)")
plt.title("Wind Speed vs. Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Wind Direction vs. Time
plt.figure(figsize=(12, 6))
for turbine_id in df_hourly['turbine_id'].unique():
    turbine_data = df_hourly[df_hourly['turbine_id'] == turbine_id]
    plt.plot(turbine_data['date_time'], turbine_data['wind_dir_adjusted'], label=f"Wind turbine {turbine_id}")

plt.xlabel("Time")
plt.ylabel("Wind direction (deg)")
plt.title("Wind Direction vs. Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

# Power Curve (Wind Speed vs. Active Power)
# Assuming active power data is available in the original files
# If not, you'll need to load it similarly to wind speed and direction
plt.figure(figsize=(12, 6))
for turbine_id in df_hourly['turbine_id'].unique():
    turbine_data = df_hourly[df_hourly['turbine_id'] == turbine_id]
    plt.scatter(turbine_data['wind_speed'], turbine_data['power'], label=f"Wind turbine {turbine_id}", alpha=0.5)  # Replace 'wind_speed' with active power column if available

plt.xlabel("Wind Speed (m/s)")
plt.ylabel("Active Power (kW)")
plt.title("Power Curve (Wind Speed vs. Active Power)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%
# Power Evolution (Active Power vs time)
plt.figure(figsize=(12, 6))
for turbine_id in df_hourly['turbine_id'].unique():
    turbine_data = df_hourly[df_hourly['turbine_id'] == turbine_id]
    plt.plot(turbine_data['date_time'], np.cumsum(turbine_data['power']), label=f"Wind turbine {turbine_id}")

plt.xlabel("Time")
plt.ylabel("Power (kW)")
plt.title("Power vs. Time")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()

#%%

tot_power=sum(turbine_data['power']) # GW

