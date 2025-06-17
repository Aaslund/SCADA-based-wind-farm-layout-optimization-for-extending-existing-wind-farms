# -*- coding: utf-8 -*-
"""
Created on Sat May 24 18:06:47 2025

@author: erica
"""


# AEP validation 3

import pandas as pd
import numpy as np
import os

from pathlib import Path
import sys
# Get the parent directory (A) and add it to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

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
   
#%%
n = 12
ang_interval = 360/n
# Define custom bin edges centered around the desired intervals
bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
bin_labels = np.arange(0.0, 360.0, ang_interval)
u_step=0.1
#%%
# Create folder path (in current directory)
load_folder = 'D:/Thesis/Data/Penmanshiel/2025-06-01/'
season_txt = 'season'
aep_tot_results=[]
f_list=[]
df = pd.read_csv(f'{load_folder}df_energy_tot.csv')
seasonal_metadata = pd.read_csv(f'{load_folder}/turbine_year_metadata.csv')

wd_count = pd.read_csv(f'{load_folder}/{n}_wind_direction_seasonal_counts.csv')
wd_count_seasonal = wd_count.groupby(['direction_bin', 'season'])[['initial_len', 'active_len']].sum().reset_index()

start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2020-12-31 23:50:00')

save_folder = load_folder
os.makedirs(save_folder, exist_ok=True)

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

u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,25.1,30]
power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050,0,0]
ct = [0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.30, 0.24, 0.19, 0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05,0,0]
my_wt = WindTurbine(name='my_wt',diameter=diam,
                    hub_height=hub_height,
                    powerCtFunction=PowerCtTabular(u,power,'W',ct))#,method='pchip'))

#%%

df['date_time'] = pd.to_datetime(df['date_time'])
df['month'] = df['date_time'].dt.month
df['season'] = df['month'].apply(get_season)


time_mask = (df['date_time'] >= start_date) & (df['date_time'] <= end_date)
start_date_data = df['date_time'].min()
end_date_data = df['date_time'].max()
if (start_date_data != start_date) |(end_date_data != end_date):
    print("Warning start or end date not matching between data and script")
    
# Apply the mask to your DataFrame
df = df[time_mask]
tot_year_fraction=((end_date_data-start_date_data).days+1)/365

weibull_results = pd.read_csv(f'{load_folder}/Weibull/{n}_winddirections/{season_txt}/weibull_results_tot.csv')
season_list=['Winter', 'Spring', 'Summer', 'Autumn']
aep_real_list=[]
aep_pywake_list=[]
aep_tot_results = []
aep_real_tot = 0
aep_pywake_tot = 0
aep_pywake_adj_tot = 0

for season_i in season_list:
    df_season = df[df['season'] == season_i].copy()
    df_season = df_season[(df_season['wind_dir'] >= 0) & (df_season['wind_dir'] <= 360)]
    df_season.loc[:, 'wind_dir'] = df_season['wind_dir'].apply(lambda wd: wd - 360 if wd >= 360 - ang_interval / 2 else wd)
    df_season['wind_dir_bin'] = pd.cut(df_season['wind_dir'], bins=bin_edges, labels=bin_labels, right=False).astype(float)

    # Real AEP
    aep_real = df_season['energy'].sum() * 1e-6 / tot_year_fraction

    # Seasonal metadata
    seasonal_metadata_i = seasonal_metadata[seasonal_metadata['season'] == season_i]
    initial_len = seasonal_metadata_i['initial_len'].sum()
    active_len = seasonal_metadata_i['active_len'].sum()

    # Weibull and frequency parameters for season
    weibull_results_season = weibull_results[weibull_results['season'] == season_i]
    f = np.array(weibull_results_season['wind_dir_probability'])
    f[-1] = 1 - sum(f[0:-1])
    A = np.array(weibull_results_season['A_mle'])
    k = np.array(weibull_results_season['k_mle'])
    wd = bin_labels
    ti =  np.array(weibull_results_season['TI'])

    # Setup PyWake site and model
    site = XRSite(
        ds=xr.Dataset(data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ('wd', ti)},
                      coords={'wd': wd}),
        initial_position=np.array([x, y]).T
    )

    wdm = ZongGaussianDeficit(use_effective_ws=True, rotorAvgModel=GaussianOverlapAvgModel())
    wfm = PropagateUpDownIterative(site, my_wt, wdm,
                                    blockage_deficitModel=Rathmann(use_effective_ws=True),
                                    turbulenceModel=STF2017TurbulenceModel())

    sim_res = wfm(x, y, type=0, wd=wd, ws=np.arange(u[0], u[-2], u_step))
    aep_pywake = float(sim_res.aep().isel(wt=np.arange(len(x))).sum()) / 4*1e3

    # Wind-direction-wise results
    for i in range(len(wd)):
        wd_i = wd[i]
        mask = df_season[df_season['wind_dir_bin'] == wd_i]
        aep_real_i = mask['energy'].sum() * 1e-6 / tot_year_fraction

        aep_pywake_i = float(sim_res.aep().isel(wd=i).sum()) / 4*1e3

        # Get seasonal directional metadata for this bin
        mask2 = wd_count_seasonal[
            (wd_count_seasonal['direction_bin'] == wd[i]) &
            (wd_count_seasonal['season'] == season_i)
        ]

        output_frac = mask2['active_len'].sum() / mask2['initial_len'].sum()

        aep_pywake_adj_i = aep_pywake_i * output_frac

        aep_tot_results.append({
            'season': season_i,
            'wind_dir': wd_i,
            'prob': round(f[i] * 100, 2),
            'A': round(A[i], 2),
            'k': round(k[i], 2),
            'aep_real': aep_real_i,
            'aep_pywake': aep_pywake_i,
            'aep_pywake_adj': aep_pywake_adj_i,
            'error': aep_pywake_adj_i - aep_real_i,
            'relative error': abs(aep_pywake_adj_i - aep_real_i) / aep_real_i * 100 if aep_real_i > 0 else 0,
            'overestimated': str(aep_pywake_adj_i > aep_real_i)
        })

    # Season summary
    mask3 = wd_count_seasonal[
        (wd_count_seasonal['season'] == season_i)
    ]
    output_frac_season = mask3['active_len'].sum() / mask3['initial_len'].sum()
    aep_pywake_adj = aep_pywake * output_frac_season

    aep_tot_results.append({
        'season': season_i,
        'wind_dir': 'Total',
        'aep_real': aep_real,
        'aep_pywake': aep_pywake,
        'aep_pywake_adj': aep_pywake_adj,
        'error': aep_pywake_adj - aep_real,
        'relative error': abs(aep_pywake_adj - aep_real) / aep_real * 100 if aep_real > 0 else 0,
        'overestimated': str(aep_pywake_adj > aep_real)
    })

    aep_real_tot += aep_real
    aep_pywake_tot += aep_pywake
    aep_pywake_adj_tot += aep_pywake_adj

# Final overall summary
aep_tot_results.append({
    'season': 'Total',
    'wind_dir': 'Total',
    'aep_real': aep_real_tot,
    'aep_pywake': aep_pywake_tot,
    'aep_pywake_adj': aep_pywake_adj_tot,
    'error': aep_pywake_adj_tot - aep_real_tot,
    'relative error': abs(aep_pywake_adj_tot - aep_real_tot) / aep_real_tot * 100,
    'overestimated': str(aep_pywake_adj_tot > aep_real_tot)
})

# Save results
aep_tot_results_df = pd.DataFrame(aep_tot_results)
aep_tot_results_df.to_csv(f'{save_folder}/wind_aep_results_{n}winddirections_{season_txt}.csv', index=False)
