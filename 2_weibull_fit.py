# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:06:37 2025

@author: erica
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from support_functions.weibull_parameter_estimation import weibull_pdf
from support_functions.weibull_parameter_estimation import estimate_weibull_mle
from support_functions.weibull_parameter_estimation import evaluate_weibull_fit
from support_functions.weight_function import weight_function
from windrose import WindroseAxes
import matplotlib.cm as cm


plt.rcParams.update({
    'font.size': 14,           # Default font size for text
    'axes.titlesize': 16,      # Title font size
    'axes.labelsize': 14,      # Axis label font size
    'xtick.labelsize': 12,     # X-axis tick label size
    'ytick.labelsize': 12,     # Y-axis tick label size
    'legend.fontsize': 12,     # Legend font size
})
#%%  === 3. Angle Bins
season = 0
n = 12
wake_angle = 10
ang_interval = 360/n
# Define custom bin edges centered around the desired intervals
bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
bin_labels = np.arange(0.0, 360.0, ang_interval)   # Labels corresponding to bin centers

#%% Wind rose for the whole data
wind_speed_bins = [0, 3, 7, 11, 15]
colors = cm.viridis(np.linspace(0, 1, len(wind_speed_bins)))
fig, ax = plt.subplots(figsize=(9, 8), subplot_kw=dict(projection='windrose'))

ax.bar(df_hourly['wind_dir'], df_hourly['wind_speed'], normed=True, bins=wind_speed_bins, colors=colors, opening=0.8, edgecolor='white',nsector=12)
ax.legend(title='Wind Speed (m/s)',bbox_to_anchor=(1.2, 0.5),loc='center left', title_fontsize=25, prop={'size': 20})
ax.set_title(f'Wind Rose with all points', pad=40, fontsize=20)
plt.tight_layout()
#%% Iterate over wind direction bins
# Initialize an empty list to store filtered DataFrames
filtered_dfs = []
wind_speed_median = []
k_mle_lst = []
c_mle_lst = []

#%% With seasons
if season==1:
    # Define a function to map months to seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        elif month in [9, 10, 11]:
            return 'Autumn'
    season_list=['Winter', 'Spring', 'Summer', 'Autumn']
else:
    def get_season(month):
        return 'no season'
    season_list=['no season']
    # Add a 'season' column
    df_hourly['season'] = df_hourly['month'].apply(get_season)

# Initialize lists to store results
seasonal_results = []
filtered_dfs = []

# Define wind speed bins
wind_speed_bins = [0, 3, 7, 11, 15]
wind_speed_labels = ['0-3 m/s', '3-7 m/s', '7-11 m/s', '11-15 m/s', '>15 m/s']
colors = cm.viridis(np.linspace(0, 1, len(wind_speed_bins)))

wind_dir_results = {}

# Iterate over seasons
for season in season_list:
    # Filter data for the current season
    df_season = df_hourly[df_hourly['season'] == season]
    total_points_season = 0 
    # Iterate over wind direction bins
    for wind_dir_i in bin_labels:
        # Compute weights for the current wind direction bin center
        weight = weight_function(x, y, wind_dir_i, turbine_diameter, wake_angle)
        
        # Convert weight array to a boolean mask
        mask = pd.Series(weight, index=range(1, nturb+1))
        total_points_season += len(df_season[(df_season['wind_dir_bin'] == wind_dir_i) & (df_season['turbine_id'].map(mask) != 0)])
        
    for wind_dir_i in bin_labels:
        all_wind_speeds = []
        all_weights = []
        # Compute weights for the current wind direction bin center
        weight = weight_function(x, y, wind_dir_i, turbine_diameter, wake_angle)
        
        # Convert weight array to a boolean mask
        mask = pd.Series(weight, index=range(1, nturb+1))
        # Filter df_season for the current wind direction bin and apply the weight mask
        df_filtered = df_season[(df_season['wind_dir_bin'] == wind_dir_i) & (df_season['turbine_id'].map(mask) != 0)]
        
        wind_dir_probability = len(df_filtered)/total_points_season
        
        # Perform Weibull fitting if the filtered dataframe is not empty
        if not df_filtered.empty:
            filtered_dfs.append(df_filtered)
            k_mle, A_mle = estimate_weibull_mle(df_filtered['wind_speed'])
            wind_speed_median = df_filtered['wind_speed'].median()
            
            # Store results
            seasonal_results.append({
                'season': season,
                'wind_dir_bin': wind_dir_i,
                'k_mle': k_mle,
                'A_mle': A_mle,
                'wind_speed_median': wind_speed_median,
                'wind_dir_probability': wind_dir_probability
            })
            
            # Plot the Weibull fit
            nb_bins = 50
            plt.figure()
            evaluate_weibull_fit(df_filtered['wind_speed'], k_mle, A_mle, f"MLE ({season}, {wind_dir_i}°, {len(df_filtered['wind_speed'])} points), prob: {round(wind_dir_probability*100,2)}", nb_bins)
# Convert results to a DataFrame for easier analysis
df_total_results = pd.DataFrame(seasonal_results)

# Concatenate all filtered DataFrames into one
df_final = pd.concat(filtered_dfs, ignore_index=True)

if season==1:    
    #%% Wind rose for each season data
    wind_speed_bins = [0, 3, 7, 11, 15]
    colors = cm.viridis(np.linspace(0, 1, len(wind_speed_bins)))

    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
        fig, ax = plt.subplots(subplot_kw=dict(projection='windrose'))
        season_data = df_final[(df_final['season'] == season)]    
        ax.bar(season_data['wind_dir'], season_data['wind_speed'], normed=True, bins=wind_speed_bins, colors=colors, opening=0.8, edgecolor='white',nsector=12)
        ax.set_rmax(rmax=22)
        ax.set_legend(title='Wind Speed (m/s)')
        ax.set_title(f"Wind Rose for {season} with {len(season_data['wind_speed'])} points)")

#%% Wind rose for the filtered data
wind_speed_bins = [0, 3, 7, 11, 15]
colors = cm.viridis(np.linspace(0, 1, len(wind_speed_bins)))
fig, ax = plt.subplots(figsize=(9, 8),subplot_kw=dict(projection='windrose'))

ax.bar(df_final['wind_dir'], df_final['wind_speed'], normed=True, bins=wind_speed_bins, colors=colors, opening=0.8, edgecolor='white',nsector=12)
ax.legend(title='Wind Speed (m/s)',bbox_to_anchor=(1.2, 0.5),loc='center left', title_fontsize=25, prop={'size': 20})
ax.set_title(f"Wind Rose with {round(len(df_final['wind_speed'])/npoints_init*100,2)} % of the data", pad=40, fontsize=20)
