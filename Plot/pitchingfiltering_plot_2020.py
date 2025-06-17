# -*- coding: utf-8 -*-
"""
Created on Thu May  8 08:50:54 2025

@author: erica
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May  6 15:39:14 2025

@author: erica
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  1 12:48:34 2025

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

from pathlib import Path
import sys
# Get the parent directory (A) and add it to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)


n_wt = 14
cut_in_ws = 3.5
cut_out_ws = 25
nominal_ws = 14.5
rated_power=2050

from support_functions.ct_eval import ct_eval
plt.rcParams.update({'font.size': 15})
u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050]
ct = [0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.30, 0.24, 0.19, 0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05]

#%% Import and save
# Function to extract turbine ID and year from filename
def extract_turbine_year(filename):
    parts = os.path.basename(filename).split("_")
    turbine_id = int(parts[3])  # Extract turbine number
    year = parts[4][:4]  # Extract year
    return turbine_id, year

# Get current date and time
now = datetime.now()
date_folder = now.strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
time_str = now.strftime("%H_%M")  # Format: HH_MM

# Create folder path (in current directory)
save_folder = os.path.join('D:/Thesis/Data/Penmanshiel/', date_folder)
save_folder = save_folder + '/2020/'
# Create folder if it doesn't exist
os.makedirs(save_folder, exist_ok=True)

# Load all turbine data
parent_folder = "D:/Thesis/Data/Penmanshiel/"
all_files = glob.glob(os.path.join(parent_folder, "Penmanshiel_SCADA_2020_*", "Turbine_Data_Penmanshiel_*.csv"))
df_windspeed_list = []
df_energy_list = []
df_energyPitch_list = []
metadata_records = []


for file in all_files:
    turbine_id, year = extract_turbine_year(file)
    df = pd.read_csv(file, header=0, usecols=["# Date and time", "Wind speed (m/s)", "Wind direction (°)", "Energy Export (kWh)", "Power (kW)", "Blade angle (pitch position) A (°)"], skiprows=9, parse_dates=["# Date and time"])
    df.columns = ['date_time', 'wind_speed', 'wind_dir', 'energy', 'power', 'pitch']
    df["turbine_id"] = turbine_id if turbine_id < 3 else turbine_id - 1
    df["year"] = int(year)
    total_rows = len(df)
    
    df_energy = df.copy()
    df_energy = df_energy.dropna()
    df_energy = df_energy[(df_energy['wind_speed'] < cut_in_ws) | ((df_energy['power'] > 2.7) & (df_energy['wind_speed'] >= cut_in_ws))]
    print(f"Rows after zero and negative power removal: {len(df_energy)}")
    plt.figure(figsize=[9,3])
    plt.plot(df_energy['wind_speed'],df_energy['power'],'.',label='SCADA data',markersize=1)
    df_energyPitch = df_energy[((df_energy['pitch'] < 40)& (df_energy['wind_speed']>=nominal_ws) &(df_energy['power'] > 0.92*rated_power)) | 
                          ((df_energy['pitch'] < 7)&(df_energy['wind_speed'] <nominal_ws))|
                          (df_energy['wind_speed'] <cut_in_ws)]
    print(f"Rows after pitch removal: {len(df_energyPitch)}")
    plt.plot(df_energyPitch['wind_speed'],df_energyPitch['power'],'.',label='Pitch filtered data',markersize=1)
    plt.plot(u,power,label='Power curve')
    plt.xlabel('Wind speed (m/s)')
    plt.ylabel('Power (kW)')
    plt.legend(loc='upper left')
    plt.show()
    df_energy_list.append(df_energy)
    df_energyPitch_list.append(df_energyPitch)

#%%    
plt.figure(figsize=[9,3])
df_energy_all = pd.concat(df_energy_list, ignore_index=True)
df_energyPitch_all = pd.concat(df_energyPitch_list, ignore_index=True)
print(f"Rows after zero and negative power removal: {len(df_energy)}")
plt.plot(df_energy_all['wind_speed'],df_energy_all['power'],'.',label='SCADA data',markersize=1)
print(f"Rows after pitch removal: {len(df_energyPitch)}")
plt.plot(df_energyPitch_all['wind_speed'],df_energyPitch_all['power'],'.',label='Pitch filtered',markersize=1)
plt.xlabel('Wind speed (m/s)')
plt.ylabel('Power (kW)')
plt.legend(loc='upper left')
plt.show()

#%%
mask = (df_energyPitch['power'] < 1900)&(df_energyPitch['power'] > 1700) &(df_energyPitch['wind_speed'] >15)