# -*- coding: utf-8 -*-
"""
Created on Tue May  6 17:55:54 2025

@author: erica
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May  3 11:20:07 2025

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

from support_functions.weight_function import weight_function
from windrose import WindroseAxes
import matplotlib.cm as cm
from datetime import datetime
plt.rcParams.update({'font.size': 40})
    
def plot_windrose(df, wind_speed_bins, wind_speed_labels, save_path,nplot,rmax):
    # Definer vindhastighets-bin og farger
    df['wind_dir'] = df['wind_dir'] % 360
    #colors = cm.viridis(np.linspace(0, 1, len(wind_speed_bins)-1))

    # Lag vindrose
    fig = plt.figure(figsize=(9, 9))
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(df['wind_dir'], df['wind_speed'], bins=wind_speed_bins, normed=True,
           opening=0.8, edgecolor='white', cmap=cm.viridis, nsector=nplot)
    legend=ax.set_legend(title="Wind speed [m/s]", bbox_to_anchor=(1.47, 0), loc='lower right',prop={'size': 25},title_fontsize=25)
    # Tilpass teksten i legenden
    for t, label in zip(legend.get_texts(), wind_speed_labels):
        t.set_text(label)
        t.set_fontsize(25)
    #plt.title(title)
    # Fix radial axis scale
    ax.set_rlim((0,rmax))
    ax.set_rgrids([rmax/5,2*rmax/5,3*rmax/5,4*rmax/5,rmax],
                  [str(rmax/5),str(2*rmax/5),str(3*rmax/5),str(4*rmax/5),str(rmax)],
                  fontsize=30)
                   
    ax.tick_params(labelsize=30)
    
    angles = np.linspace(0, 360, 8, endpoint=False)  # angles for the wind directions
    labels = ['N', 'NE', 'E', 'SE','S','SW', 'W','NW']
    
    ax.set_thetagrids(angles, labels, fontsize=30)
    
    # Shift the direction labels outward
    for label in ax.get_xticklabels():
        label.set_y(label.get_position()[1] - 0.1)
    frequencies = ax._info['table']  # matrix of shape (len(bins)-1, n_sectors)
    #wind_dir_bins = [f"{d}" for d in ax._info['dir']]  # e.g. [0, 22.5, ..., 337.5]
    bin_centers = np.arange(0, 360, 360 // nplot)
    wind_dir_bins = [f"{d}°" for d in bin_centers]
    wind_speed_ranges = wind_speed_labels
    
    csv_path = save_path.replace('.png', '.csv')
    df_out = pd.DataFrame(frequencies, index=wind_speed_ranges, columns=wind_dir_bins)
    df_out.loc['Total'] = df_out.sum(axis=0)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_out.to_csv(csv_path)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    plt.close()
    
    

# Define wind speed bins
wind_speed_bins = [0, 3.5, 7, 11, 15]
wind_speed_labels = ['0-3.5 m/s', '3.5-7 m/s', '7-11 m/s', '11-15 m/s', '>15 m/s']
#colors = cm.viridis(np.linspace(0, 1, len(wind_speed_bins)))

plt.rcParams.update({'font.size': 20})
#%%  === 3. Angle Bins
season_bool = 0
n = 12
nplot = 12
wake_angle = 20
ang_interval = 360/n
# Define custom bin edges centered around the desired intervals
bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
bin_labels = np.arange(0.0, 360.0, ang_interval)   # Labels corresponding to bin centers
if nplot==12:
    rmax=23
elif nplot==36:
    rmax=9
else:
    raise ValueError("Define a rmax")
#%%
load_folder = 'D:/Thesis/Data/Penmanshiel/2025-05-27/'
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

#%% === LAGREMAPPE ===
today = datetime.now().strftime("%Y-%m-%d")
figure_folder = f'D:/Thesis/Figures/Penmanshiel/Wind rose/{today}/'
os.makedirs(figure_folder, exist_ok=True)
# === PLOT before WEIGHTS ===
plot_windrose(df_hourly, wind_speed_bins, wind_speed_labels, os.path.join(figure_folder, f'windrose_before_weights{n}_winddirections_{nplot}nplot.png'),nplot,rmax)

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
        df_filtered = df_season[(df_season['wind_dir_bin'] == wind_dir_i) & (df_season['turbine_id'].map(mask) != 0)]
        filtered_dfs.append(df_filtered)
# === PLOTT ETTER WEIGHTS ===
df_weighted = pd.concat(filtered_dfs, ignore_index=True)

df_weighted['wind_dir'] = df_weighted['wind_dir'] % 360
plot_windrose(df_weighted, wind_speed_bins, wind_speed_labels, os.path.join(figure_folder, f'windrose_after_weights_{n}_winddirections_{nplot}nplot.png'),nplot,rmax)