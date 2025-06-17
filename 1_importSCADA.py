# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:24:10 2025

@author: erica
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

from scipy.stats import zscore

from haversine import haversine
from expand_param import expand_param
from ct_eval import ct_eval

from py_wake.wind_turbines import WindTurbine, WindTurbines
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
#%% # Load the Excel file
file_path = "D:/Thesis/Data/Penmanshiel/Penmanshiel_WT_static.csv"
df = pd.read_csv(file_path)

# Extract columns into separate arrays
latitude = df["Latitude"].to_numpy()
longitude = df["Longitude"].to_numpy()
# Find the index of the first NaN element
nan_indices = np.where(np.isnan(latitude))[0]  # Get indices of NaNs
nturb = nan_indices[0] if nan_indices.size > 0 else len(latitude)  # Get first NaN index


ref_lat = 55.90221379420022 
ref_lon = -2.309024986090815


# there is no turbine 3
x = np.zeros(nturb)
y = np.zeros(nturb)
for i in range(nturb):
    x[i]=haversine(ref_lat, ref_lon, ref_lat, longitude[i])
    y[i]=haversine(ref_lat, ref_lon, latitude[i], ref_lon)

hub_height = df["Hub Height (m)"].to_numpy()
manufacturer = df["Manufacturer"].to_list()
model = df["Model"].to_list()
diam = df["Rotor Diameter (m)"].to_numpy()

#%% turbine info
rated_wind_speed = 14.5
yaw_angle = 0

# Create turbine_id list like WTG001, WTG002, ..., WTG00N
turbine_id = [i for i in range(1, nturb + 1)]

rotor_diameter = diam[:nturb]
name={manufacturer[i]+" "+model[i]}
# Build DataFrame
turbine_info = pd.DataFrame({
    'turbine_id': turbine_id,
    'rated_wind_speed': expand_param(rated_wind_speed, nturb),
    'rotor_diameter': expand_param(rotor_diameter, nturb),
    'yaw_angle': expand_param(yaw_angle, nturb),
    'x': x,
    'y': y,
    'name': expand_param(name,nturb)
})


#%% Power curve
u = np.concatenate(([0.1], [3.5], np.arange(4, 14.5, 0.5), [25], [25.1], [35]))
power = [0, 0.1, 55,110, 186, 264, 342, 424, 506, 618, 730, 865, 999, 1195, 1391,
         1558, 1724, 1829, 1909, 1960, 2002, 2025, 2044, 2050,0,0]
ct, power_interp, u_highres = ct_eval(u,power,rho=1.225)
#%% Make turbine list and plot wind farm
wt_lst = [None]*nturb
for i in range(nturb):
    my_wt = WindTurbine(name,
                    diameter=int(diam[i]),
                    hub_height=int(hub_height[i]),
                    powerCtFunction=PowerCtTabular(u,power,'kW',ct,method='pchip'))
    wt_lst[i] = my_wt
wts = WindTurbines.from_WindTurbine_lst(wt_lst)
#%%
wts.plot_xy(x,y,)
plt.xlim(-200,2000)
plt.ylim(-500,800)
plt.xlabel('x [m]')
plt.ylabel('y [m]')
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
plt.legend()

#%%  === Load and Combine SCADA Data ===
parent_folder = "D:/Thesis/Data/Penmanshiel/"
all_files = glob(os.path.join(parent_folder, "*", "Turbine_Data_Penmanshiel_*.csv"))

def extract_turbine_year(filename):
    parts = os.path.basename(filename).split("_")
    turbine_id = int(parts[3])  # Extract turbine number
    year = parts[4][:4]  # Extract year
    return turbine_id, year

df_list = []
for file in all_files:
    turbine_id, year = extract_turbine_year(file)
    df = pd.read_csv(file, header=0, usecols=["# Date and time", "Wind speed (m/s)", "Wind direction (°)"],skiprows=9, parse_dates=["# Date and time"])
    df.columns = ['date_time', 'wind_speed', 'wind_dir']
    # df["turbine_id"] = turbine_id
    df["turbine_id"] = turbine_id if turbine_id < 3 else turbine_id - 1
    # if data from Penmanshiel
    df["year"] = int(year)
    df_list.append(df)
    
#%%
df_all = pd.concat(df_list, ignore_index=True)

#%%
df_all.set_index("date_time", inplace=True)
df_all.sort_index(inplace=True)

#%% === 1. Outlier Removal ===
df_all = df_all[(df_all['wind_speed'] > 0.5) & (df_all['wind_speed'] < 30)]
df_all = df_all[np.abs(zscore(df_all['wind_speed'])) < 3] #removal of extreme outliers, 3 standard deviations from the mean

#%% === 2. Time Averaging ===
df_hourly = df_all.groupby(['turbine_id', pd.Grouper(freq='h')]).mean().reset_index()

# Adjust wind direction so that 345°–360° is mapped to -15°–0°
# Adjust wind direction so that 345°–360° is mapped to -15°–0°
df_hourly['wind_dir_adjusted'] = df_hourly['wind_dir'].apply(lambda wd: wd - 360 if wd >= 345 else wd)

