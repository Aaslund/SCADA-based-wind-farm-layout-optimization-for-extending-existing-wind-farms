# -*- coding: utf-8 -*-
"""
Created on Thu May 29 20:36:18 2025

@author: erica
"""

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

plt.rcParams.update({'font.size': 15})
#%%  === 3. Angle Bins
season_bool = 0
wake_angle = 20
n1 = 12
ang_interval = 360/n1
# Define custom bin edges centered around the desired intervals
bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
bin_labels1 = np.arange(0.0, 360.0, ang_interval)   # Labels corresponding to bin centers

n2 = 36
ang_interval = 360/n2
# Define custom bin edges centered around the desired intervals
bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
bin_labels2 = np.arange(0.0, 360.0, ang_interval)   # Labels corresponding to bin centers


#%%

load_folder = 'D:/Thesis/Data/Penmanshiel/2025-06-01/'
if season_bool==1:
    season_txt='season'
else:
    season_txt='no_season'

#df_hourly = df_10min
weibull_results12 = pd.read_csv(
    f'{load_folder}/Weibull/12_winddirections/{season_txt}/weibull_results_tot.csv')
weibull_results36 = pd.read_csv(
    f'{load_folder}/Weibull/36_winddirections/{season_txt}/weibull_results_tot.csv')

# wind direction
wd1 = bin_labels1
# turbulence intensity
ti1 = np.array(weibull_results12['TI'])

# wind direction
wd2 = bin_labels2
# turbulence intensity
ti2 = np.array(weibull_results36['TI'])
#%% Create save folder
save_folder = r'D:\Thesis\Figures\Penmanshiel\TI\2025-06-01'
os.makedirs(save_folder, exist_ok=True)

#%% Plot 1: Wind direction vs TI
plt.figure(figsize=(9, 4))
plt.plot(wd1, ti1, '-o', markersize=8,label='12 directions')
plt.plot(wd2, ti2, '-o', markersize=8,label='36 directions',alpha=0.7)
plt.xlabel('Wind Direction [°]')
plt.ylabel('Turbulence Intensity [-]')
#plt.title('TI vs Wind Direction')
#plt.grid(True)
plt.legend(loc='lower right')
#ticks=list(wd[0:-1:4])
#plt.xticks(ticks)
plt.tight_layout()
plt.savefig(os.path.join(save_folder, 'TI_vs_WindDirection_both.png'))
plt.close()
