# -*- coding: utf-8 -*-
"""
Created on Thu May  8 08:31:57 2025

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
# Create folder path (in current directory)
load_folder = 'D:/Thesis/Data/Penmanshiel/2025-06-01/'
metadata = pd.read_csv(f'{load_folder}/turbine_year_metadata.csv')

save_folder = load_folder
os.makedirs(save_folder, exist_ok=True)
#%%
n = 12
ang_interval = 360/n
# Define custom bin edges centered around the desired intervals
bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
bin_labels = np.arange(0.0, 360.0, ang_interval)
u_step=0.1
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
aep_tot_results=[]
year_list=[2017,2018,2019,2020]
f_list=[]
aep_real_tot=0
aep_pywake_tot=0
aep_pywake_adj_tot=0
for year in year_list:
    df_10min = pd.read_csv(f'{load_folder}df_energy_all_{year}.csv')
    aep_real = df_10min['energy'].sum()*1e-6 #MWh
    if year == 2020: #leap year
        aep_real=aep_real*365/366
    weibull_results = pd.read_csv(f'{load_folder}/Weibull/12_winddirections/no_season/weibull_results_{year}.csv')
    # frequency of the different wind directions
    f = np.array(weibull_results['wind_dir_probability'])
    f /= f.sum()
    # Weibull scale parameter
    A = np.array(weibull_results['A_mle'])
    # Weibull shape parameter
    k = np.array(weibull_results['k_mle'])
    # wind direction
    wd = bin_labels
    # turbulence intensity
    ti = np.array(weibull_results['TI'])
    #%% Make site
    site = XRSite(
        ds=xr.Dataset(data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ('wd', ti)},
                      coords={'wd': wd}), initial_position=np.array([x, y]).T)
    
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
                ws=np.arange(u[0], u[-2], u_step)    # Wind speed
                )
    
    aep_pywake = float(sim_res.aep().isel(wt=np.arange(len(x))).sum())*1e3
    metadata_year = metadata[metadata['year']==year]
    output_frac = float(metadata_year['active_len'].sum()/metadata_year['initial_len'].sum())
    aep_pywake_adj = aep_pywake*output_frac
    aep_tot_results.append({
        'year': year,
        'aep_real' : f"{aep_real:.2f}",
        'aep_pywake' : f"{aep_pywake:.2f}",
        'aep_pywake_adj': f"{aep_pywake_adj:.2f}",
        'error':f"{aep_pywake_adj-aep_real:.2f}",
        'relative error':f"{abs(aep_pywake_adj-aep_real)/aep_real*100:.2f}",
        'overestimated':str(aep_pywake_adj>aep_real)
        })
    aep_real_tot=aep_real_tot+aep_real
    aep_pywake_tot=aep_pywake_tot+aep_pywake
    aep_pywake_adj_tot=aep_pywake_adj_tot+aep_pywake_adj

aep_tot_results.append({
    'year': 'Total',
    'aep_real' : f"{aep_real_tot:.2f}",
    'aep_pywake' : f"{aep_pywake_tot:.2f}",
    'aep_pywake_adj': f"{aep_pywake_adj_tot:.2f}",
    'error': f"{aep_pywake_adj_tot-aep_real_tot:.2f}",
    'relative error': f"{abs(aep_pywake_adj_tot-aep_real_tot)/aep_real_tot*100:.2f}",
    'overestimated':str(aep_pywake_adj_tot>aep_real_tot)
    })   

aep_tot_results.append({
    'year': 'Averaged yearly',
    'aep_real' : f"{aep_real_tot/4:.2f}",
    'aep_pywake' : f"{aep_pywake_tot/4:.2f}",
    'aep_pywake_adj': f"{aep_pywake_adj_tot/4:.2f}",
    'error': f"{aep_pywake_adj_tot/4-aep_real_tot/4:.2f}",
    'relative error': f"{abs(aep_pywake_adj_tot-aep_real_tot)/aep_real_tot*100:.2f}",
    'overestimated':str(aep_pywake_adj_tot>aep_real_tot)
    })        
#%%            # Convert results to a DataFrame for easier analysis
aep_tot_results = pd.DataFrame(aep_tot_results)
aep_tot_results.to_csv(f'{save_folder}/aep_results_{n}winddirections_year.csv')