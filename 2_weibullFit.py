# -*- coding: utf-8 -*-
"""
Created on Tue May  6 17:55:54 2025

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

from support_functions.weibull_parameter_estimation2 import estimate_weibull_mle2
from support_functions.weibull_parameter_estimation2 import evaluate_weibull_fit2
from support_functions.weight_function import weight_function
from datetime import datetime
from support_functions.plot_weibull_fit import plot_weibull_fit

plt.rcParams.update({'font.size': 15})
#%%  === 3. Angle Bins
season_bool = 0
n = 36
wake_angle = 20
ang_interval = 360/n
# Define custom bin edges centered around the desired intervals
bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
bin_labels = np.arange(0.0, 360.0, ang_interval)   # Labels corresponding to bin centers

#%%
load_folder = 'D:/Thesis/Data/Penmanshiel/2025-06-01/'
df_hourly = pd.read_csv(f'{load_folder}df_windspeed_tot.csv')
start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2020-12-31 23:00:00') # because of hourly average
df_hourly['date_time'] = pd.to_datetime(df_hourly['date_time'])
time_mask = (df_hourly['date_time'] >= start_date) & (df_hourly['date_time'] <= end_date)
start_date_data = df_hourly['date_time'].min()
end_date_data = df_hourly['date_time'].max()
if (start_date_data != start_date) |(end_date_data != end_date):
    print("WARNING start or end date not matching between data and script")
# Apply the mask to your DataFrame
df_hourly = df_hourly[time_mask]
#%%
now = datetime.now()
date_folder = now.strftime("%Y-%m-%d")  # Format: YYYY-MM-DD
time_str = now.strftime("%H_%M")  # Format: HH_MM
figure_folder = f'D:/Thesis/Figures/Penmanshiel/Weibull/{date_folder}/old/{n}_winddirections/'
if season_bool:
    figure_folder = figure_folder+'season/'
else:
    figure_folder = figure_folder+'no_season/'

figure_folder =f"{figure_folder}/tot/ref/"
os.makedirs(figure_folder, exist_ok=True)

font=15
figure_folder15 = figure_folder+f'/{font}font/'
os.makedirs(figure_folder15, exist_ok=True)

font=17
figure_folder20 = figure_folder+f'/{font}font/'
os.makedirs(figure_folder20, exist_ok=True)

figure_folderZoom = figure_folder+f'/Zoom/'
os.makedirs(figure_folderZoom, exist_ok=True)
#%%
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
rated_wind_speed = 14.5
#%% Iterate over wind direction bins
# Initialize an empty list to store filtered DataFrames
filtered_dfs = []
wind_speed_median = []
k_mle_lst = []
c_mle_lst = []

#%% With seasons
df_hourly.loc[:,'wind_dir'] = df_hourly['wind_dir'].apply(lambda wd: wd - 360 if wd >= 360-ang_interval/2 else wd)
df_hourly['wind_dir_bin'] = pd.cut(df_hourly['wind_dir'], bins=bin_edges, labels=bin_labels, right=False).astype(float)

if season_bool==1:
    season_list=['Winter', 'Spring', 'Summer', 'Autumn']
else:
    season_list=['no season']
    df_hourly['season']='no season'
# Initialize lists to store results
seasonal_results = []
filtered_dfs = []

wind_dir_results = {}

# Iterate over seasons
for season in season_list:
    # Filter data for the current season
    df_season = df_hourly[df_hourly['season'] == season]
    total_points_season = 0 
    # Iterate over wind direction bins  

    for wind_dir_i in bin_labels:
        # Compute weights for the current wind direction bin center
        weight = weight_function(x, y, wind_dir_i, diam, wake_angle)
        
        # Convert weight array to a boolean mask
        mask = pd.Series(weight, index=range(1, n_wt+1))
        df_bin=df_season[(df_season['wind_dir_bin'] == wind_dir_i) & (df_season['turbine_id'].map(mask) != 0)]
        total_points_season += df_bin['date_time'].nunique()      
    for wind_dir_i in bin_labels:
        all_wind_speeds = []
        all_weights = []
        # Compute weights for the current wind direction bin center
        weight = weight_function(x, y, wind_dir_i, diam, wake_angle)
        # Convert weight array to a boolean mask
        mask = pd.Series(weight, index=range(1, n_wt+1))
        # Filter df_season for the current wind direction bin and apply the weight mask
        df_filtered = df_season[(df_season['wind_dir_bin'] == wind_dir_i) & (df_season['turbine_id'].map(mask) != 0)]
        df_valid=df_filtered
        # # Count available turbines per timestamp
        # count_per_timestamp = df_filtered.groupby('date_time')['turbine_id'].nunique()
        # #print(f'count_per_timestamp:{len(count_per_timestamp)}')
        # valid_timestamps = count_per_timestamp[count_per_timestamp >= 0.3*sum(mask)].index
        # #print(f'valid_timestamps:{len(valid_timestamps)}')
        # df_valid = df_filtered[df_filtered['date_time'].isin(valid_timestamps)]
        df_filtered_1stamp = df_valid.groupby('date_time')[['wind_speed', 'TI']].median()
        wind_dir_probability = df_valid['date_time'].nunique()/total_points_season
        
        # Perform Weibull fitting if the filtered dataframe is not empty
        if not df_filtered.empty:
            filtered_dfs.append(df_filtered_1stamp)
            k_mle, loc_mle, A_mle = estimate_weibull_mle2(df_filtered_1stamp['wind_speed'])
            wind_speed_median = df_filtered_1stamp['wind_speed'].median()
            # Plot the Weibull fit
            plt.figure()
            ks,p_value, rmse, tail_error = evaluate_weibull_fit2(df_filtered_1stamp['wind_speed'], k_mle, loc_mle, A_mle, season, wind_dir_i,rated_wind_speed)

            # Store results
            seasonal_results.append({
                'season': season,
                'wind_dir_bin': wind_dir_i,
                'k_mle': k_mle,
                'A_mle': A_mle,
                'nb_time_points': len(df_filtered_1stamp),
                'wind_dir_probability':wind_dir_probability,
                'wind_speed_median': wind_speed_median,
                'ks': ks,
                'rmse': rmse,
                'TI': df_filtered_1stamp['TI'].mean(),
                'Season': season,
                'Wind direction': f"{wind_dir_i}",
                'Probability': f"{wind_dir_probability*100:.2f}",
                'k': f"{k_mle:.2f}",
                'A': f"{A_mle:.2f}",
                'K-S': f"{ks:.4f}",
                'RMSE': f"{rmse:.4f}",
                'tail error': f"{tail_error:.4f}",
                'points': df_valid['date_time'].nunique(),
                'tot_season':total_points_season,
            })
            
            figsize=(8,6)
            font = 17
            filename=f"{season}_wd{int(wind_dir_i)}.png"
            save_path = os.path.join(figure_folder15, filename)
            plot_weibull_fit(df_filtered_1stamp, k_mle, A_mle, season, wind_dir_i,
                 font_size=font, figsize=figsize, save_path=save_path)
            
            figsize=(8,6)
            font = 17
            filename=f"{season}_wd{int(wind_dir_i)}.png"
            save_path = os.path.join(figure_folder20, filename)
            plot_weibull_fit(df_filtered_1stamp, k_mle, A_mle, season, wind_dir_i,
                 font_size=font, figsize=figsize, save_path=save_path)
            plt.close()
            
            figsize=(8,6)
            font = 17
            filename=f"{season}_wd{int(wind_dir_i)}.png"
            save_path = os.path.join(figure_folderZoom, filename)
            plot_weibull_fit(df_filtered_1stamp, k_mle, A_mle, season, wind_dir_i,
                 font_size=font, figsize=figsize, save_path=save_path, xlimit=[15,25], ylimit=[0,0.025])
            plt.close()
            # Convert results to a DataFrame for easier analysis
#%%
df_total_results = pd.DataFrame(seasonal_results)
save_folder = load_folder+f'/Weibull/{n}_winddirections/'

if season_bool:
    save_folder = save_folder+'season/'
else:
    save_folder = save_folder+'no_season/'
os.makedirs(save_folder, exist_ok=True)
df_total_results.to_csv(f'{save_folder}/weibull_results_tot.csv')
