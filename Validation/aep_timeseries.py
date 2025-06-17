# -*- coding: utf-8 -*-
"""
Created on Thu May 29 11:33:53 2025

@author: erica
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path

# Get the parent directory (A) and add it to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from support_functions.weibull_parameter_estimation2 import weibull_pdf2
from support_functions.weibull_parameter_estimation2 import estimate_weibull_mle2
from support_functions.weibull_parameter_estimation2 import evaluate_weibull_fit2
from support_functions.weight_function import weight_function
from windrose import WindroseAxes
import matplotlib.cm as cm
from datetime import datetime


from pathlib import Path
import sys
# Get the parent directory (A) and add it to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)


#from support_functions.ct_eval import ct_eval

from py_wake.deficit_models import ZongGaussianDeficit
from py_wake.deficit_models import Rathmann
from py_wake.rotor_avg_models import GaussianOverlapAvgModel
from py_wake.turbulence_models import STF2017TurbulenceModel
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_farm_models import PropagateUpDownIterative

from py_wake.site import XRSite
from py_wake.site.shear import PowerShear
import xarray as xr


plt.rcParams.update({'font.size': 20})
#%%  === 3. Angle Bins
season_bool = 0
n = 12
wake_angle = 20
ang_interval = 360/n
# Define custom bin edges centered around the desired intervals
bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
bin_labels = np.arange(0.0, 360.0, ang_interval)   # Labels corresponding to bin centers


#%%

load_folder = 'D:/Thesis/Data/Penmanshiel/2025-05-30/'
df_10min = pd.read_csv(f'{load_folder}df_energy_tot.csv')
start_date = pd.to_datetime('2017-09-03')
end_date = pd.to_datetime('2017-09-28 23:50:00')
df_10min['date_time'] = pd.to_datetime(df_10min['date_time'])
time_mask = (df_10min['date_time'] >= start_date) & (df_10min['date_time'] <= end_date)
df_10min = df_10min[time_mask]
df_10min = df_10min.reset_index()

df_hourly = pd.read_csv(f'{load_folder}df_windspeed_tot.csv')
df_hourly['date_time'] = pd.to_datetime(df_hourly['date_time'])
time_mask = (df_hourly['date_time'] >= start_date) & (df_hourly['date_time'] <= end_date)
df_hourly = df_hourly[time_mask]
df_hourly = df_hourly.set_index('date_time')

#df_hourly = df_10min


df_hourly.loc[:,'wind_dir'] = df_hourly['wind_dir'].apply(lambda wd: wd - 360 if wd >= 360-ang_interval/2 else wd)
df_hourly['wind_dir_bin'] = pd.cut(df_hourly['wind_dir'], bins=bin_edges, labels=bin_labels, right=False).astype(float)

x=np.array([   0.        ,  319.22363416,  230.58157057,  500.06092411,
        774.34016385, 1030.60424373,  565.70093313,  909.048672  ,
       1172.16973239, 1497.19062847,  960.16434834, 1278.01658368,
       1542.1350215 , 1811.48969781])
y=np.array([   0.        , -277.32014705,  382.62174258,   88.0663819 ,
       -172.46333123, -418.20411911,  601.89813793,  276.65297749,
         58.93331112, -183.47162896,  689.51974012,  503.04584814,
        283.32467309,   -4.33660214])


hub_height = 90
diam = 82
n_wt=14

total_points=0
df_filtered_all=[]
for wind_dir_i in bin_labels:
    # Compute weights for the current wind direction bin center
    weight = weight_function(x, y, wind_dir_i, diam, wake_angle)
    
    # Convert weight array to a boolean mask
    mask = pd.Series(weight, index=range(1, n_wt+1))
    total_points += len(df_hourly[(df_hourly['wind_dir_bin'] == wind_dir_i) & (df_hourly['turbine_id'].map(mask) != 0)])
    
for wind_dir_i in bin_labels:
    all_wind_speeds = []
    all_weights = []
    # Compute weights for the current wind direction bin center
    weight = weight_function(x, y, wind_dir_i, diam, wake_angle)
    # Convert weight array to a boolean mask
    mask = pd.Series(weight, index=range(1, n_wt+1))
    # Filter df_hourly for the current wind direction bin and apply the weight mask
    df_filtered = df_hourly[(df_hourly['wind_dir_bin'] == wind_dir_i) & (df_hourly['turbine_id'].map(mask) != 0)]
    df_filtered_all.append(df_filtered)

df_filtered_df = pd.concat(df_filtered_all)
numeric_cols = df_filtered_df.select_dtypes(include=['number']).columns
# Group by 'date_time' and take the median only of numeric columns
df_filtered_df_grouped = df_filtered_df.groupby('date_time')[numeric_cols].median()
# Time series input variables
wd = df_filtered_df_grouped['wind_dir'].values
ws = df_filtered_df_grouped['wind_speed'].values
ti = df_filtered_df_grouped['TI'].clip(upper=0.5).values
time_stamp = df_filtered_df_grouped.index
#%%
# Create time stamps in days since start
time = len(df_filtered_df_grouped.index - df_filtered_df_grouped.index[0])/ (3600 * 24)
u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,25.1,30]
power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050,0,0]
ct = [0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.30, 0.24, 0.19, 0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05,0,0]
my_wt = WindTurbine(name='my_wt',diameter=diam,
                    hub_height=hub_height,
                    powerCtFunction=PowerCtTabular(u,power,'W',ct))#,method='pchip'))



# site_x, site_y = np.meshgrid(np.arange(0.1, 1000, 100), np.arange(0.1, 2000, 100))
# site_x, site_y = site_x.flatten(), site_y.flatten()
# site_time = np.arange(100)
# site_ws = np.random.uniform(3.0, 21.0, (len(site_x), len(site_y), len(site_time)))
# site_wd = np.random.uniform(0.0, 360.0, (len(site_x), len(site_y), len(site_time)))
# ds = xr.Dataset(
#     data_vars=dict(
#         WS=(["x", "y", "time"], site_ws),
#         WD=(["x", "y", "time"], site_wd),
#         TI=(["x", "y", "time"], np.ones_like(site_ws) * 0.1),  # hardcoded TI=0.1
#         P=1,  # deterministic wind resource
#     ),
#     coords=dict(
#         x=("x", site_x),
#         y=("y", site_y),
#         time=("time", site_time),
#     ),
# )
# non_uniform_ts_site = XRSite(ds)
#%%
#site = XRSite(
 #   ds=xr.Dataset(initial_position=np.array([x, y]).T))
from py_wake.site import UniformSite
site = UniformSite()
#%% Make wind deficit model and wind farm model
wdm = ZongGaussianDeficit(use_effective_ws=True,
                          rotorAvgModel=GaussianOverlapAvgModel()
                          )
wfm = PropagateUpDownIterative(site, my_wt, wdm,
                              blockage_deficitModel=Rathmann(use_effective_ws=True),
                              turbulenceModel=STF2017TurbulenceModel())

#%%
sim_res = wfm(x, y,   # wind turbine positions
            h=None,   # wind turbine heights (defaults to the heights defined in windTurbines)
            type=0,   # Wind turbine types
            wd=wd,    # Wind direction
            ws=ws,
            TI=ti,
            time=time_stamp
            )

#%%
nturbine_i=13
df_turb=df_10min[df_10min['turbine_id']==nturbine_i]
df_turb = df_turb.sort_values('date_time')
fig, ax = plt.subplots(figsize=(16, 4))

start_datetime = df_turb['date_time'].min().normalize()

# Convert sim_res.time from days to datetime
fig, ax = plt.subplots(figsize=(16, 4))

hourly_power=df_turb.resample('h', on='date_time').agg({
                'power': 'mean',
                'wind_speed': 'mean',
                'wind_dir': 'mean'}).reset_index()
# Plot actual turbine power
ax.plot(hourly_power['date_time'],hourly_power['power'] , '-', label='Measured Power')

# Plot simulated power with aligned time
ax.plot(time_stamp, sim_res.Power.sel(wt=nturbine_i-1), '-',label='Simulated Power')

#ax.set_xlim(pd.Timestamp("2017-09-03"), pd.Timestamp("2017-09-05"))

rec_sum = df_turb['energy'].sum()* 1e-3
sim_sum = float(sim_res.Power.sel(wt=nturbine_i-1).sum()) * 1e-3  # MWh
delta=(sim_sum-rec_sum)/rec_sum*100
if np.sign(delta) == -1:
    sign=''
else:
    sign='+'
ax.set_title(f"Measured Energy: {rec_sum:.2f} kWh | Simulated Energy: {sim_sum:.2f} kWh | Δ: {sign}{delta:.2f}%")
ax.set_xlabel("Date")
ax.set_ylabel("Power (W)")
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()

#%% wind speed and wind direction
fig2, ax2 = plt.subplots(figsize=(16, 4))

# Left y-axis: Wind speed
ax2.plot(time_stamp, ws, label='Wind Speed (m/s)', color='magenta')
ax2.set_ylabel("Wind Speed (m/s)", color='magenta')
ax2.tick_params(axis='y', labelcolor='magenta')

# Right y-axis: Wind direction
ax3 = ax2.twinx()
ax3.plot(time_stamp, wd, label='Wind Direction (°)', color='tab:green')
ax3.set_ylabel("Wind Direction (°)", color='tab:green')
ax3.tick_params(axis='y', labelcolor='tab:green')

# Common x-axis
#ax2.set_title("Wind Speed and Direction Over Time")
ax2.set_xlabel("Date")
fig2.tight_layout()
plt.show()

#%%
# Group measured data for all turbines
df_all = df_10min.sort_values('date_time')
hourly_power_all = df_all.resample('h', on='date_time').agg({
    'power': 'sum',  # sum power across turbines
    'wind_speed': 'mean',
    'wind_dir': 'mean'
}).reset_index()
hourly_power_all['power']=hourly_power_all['power']/6
# Create plot
fig, ax = plt.subplots(figsize=(16, 4))

# Plot measured total power
ax.plot(hourly_power_all['date_time'], hourly_power_all['power']*1e-3, '-', label='Measured Power')

# Sum simulated power across all turbines for each time
# Assumes sim_res.Power has shape (time, wt) and sim_res.time aligns with hourly timestamps
sim_total_power = sim_res.Power.sum(dim='wt')  # Sum across turbines (axis 1)
ax.plot(time_stamp, sim_total_power*1e-3, '-', label='Simulated Power')

# Calculate total energy
rec_sum = df_all['energy'].sum() * 1e-3  # kWh
rec_sum2=hourly_power_all['power'].sum()*1e-3
sim_sum = float(sim_total_power.sum()) * 1e-3     # kWh

# Compute relative difference
delta = (sim_sum - rec_sum) / rec_sum * 100
sign = '+' if delta >= 0 else ''

# Plot settings
ax.set_title(f"Measured Energy: {rec_sum:.2f} kWh | Simulated Energy: {sim_sum:.2f} kWh | Δ: {sign}{delta:.2f}%")
ax.set_xlabel("Date")
ax.set_ylabel("Power (kW)")
ax.legend(loc='upper right')
plt.tight_layout()
plt.show()
