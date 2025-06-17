# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 13:36:29 2025

@author: erica
"""

# -*- coding: utf-8 -*-
"""
Created on Thu May  8 08:31:57 2025

@author: erica
"""

# AEP validation with correction factor (yearly)

import pandas as pd
import numpy as np
import os

from pathlib import Path
import sys
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
import xarray as xr
from scipy.stats import weibull_min

#%%
load_folder = 'D:/Thesis/Data/Penmanshiel/2025-05-30/'
metadata = pd.read_csv(f'{load_folder}/turbine_year_metadata.csv')

save_folder = load_folder
os.makedirs(save_folder, exist_ok=True)

n = 12
ang_interval = 360/n
bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)
bin_labels = np.arange(0.0, 360.0, ang_interval)
u_step = 0.1

x = np.array([   0., 319.22363416, 230.58157057, 500.06092411,
                 774.34016385, 1030.60424373, 565.70093313, 909.048672,
                1172.16973239, 1497.19062847, 960.16434834, 1278.01658368,
                1542.1350215, 1811.48969781])
y = np.array([   0., -277.32014705, 382.62174258, 88.0663819,
               -172.46333123, -418.20411911, 601.89813793, 276.65297749,
                 58.93331112, -183.47162896, 689.51974012, 503.04584814,
                283.32467309, -4.33660214])
hub_height = 90
diam = 82
n_wt = 14

u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25.1, 30]
power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 0, 0]
ct = [0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.30, 0.24, 0.19, 0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0, 0]
my_wt = WindTurbine(name='my_wt', diameter=diam,
                    hub_height=hub_height,
                    powerCtFunction=PowerCtTabular(u, power, 'W', ct))

#%%
aep_tot_results = []
aep_tot_results_wd = []
year_list = [2017, 2018, 2019, 2020]
aep_real_tot = 0
aep_pywake_tot = 0
aep_pywake_adj_tot = 0

for year in year_list:
    if year == 2020:
        output_frac = 366/365
    else:
        output_frac=1
    df = pd.read_csv(f'{load_folder}df_energy_all_{year}.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])
    aep_real = df['energy'].sum() * 1e-6 /output_frac # MWh

    weibull_results = pd.read_csv(f'{load_folder}/Weibull/12_winddirections/no_season/weibull_results_{year}.csv')
    f = np.array(weibull_results['wind_dir_probability'])
    f /= f.sum()
    A = np.array(weibull_results['A_mle'])
    k = np.array(weibull_results['k_mle'])
    ti = np.array(weibull_results['TI'])
    wd = bin_labels

    site = XRSite(
        ds=xr.Dataset(data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ('wd', ti)},
                      coords={'wd': wd}), initial_position=np.array([x, y]).T)

    wdm = ZongGaussianDeficit(use_effective_ws=True, rotorAvgModel=GaussianOverlapAvgModel())
    wfm = PropagateUpDownIterative(site, my_wt, wdm,
                                    blockage_deficitModel=Rathmann(use_effective_ws=True),
                                    turbulenceModel=STF2017TurbulenceModel())

    ws_centers = np.arange(u[0], u[-2], u_step)
    ws_bins = np.append(ws_centers, ws_centers[-1] + u_step)
    n_ws = len(ws_centers)

    df.loc[:, 'wind_dir'] = df['wind_dir'].apply(lambda wd: wd - 360 if wd >= 360 - ang_interval / 2 else wd)
    df['wind_dir_bin'] = pd.cut(df['wind_dir'], bins=bin_edges, labels=bin_labels, right=False).astype(float)

    corrections = np.zeros((n, n_ws))
    for i, wdi in enumerate(wd):
        df_wdi = df[df['wind_dir_bin'] == wdi]
        hist_scada, _ = np.histogram(df_wdi['wind_speed'], bins=ws_bins, density=True)
        hist_scada /= np.sum(hist_scada)
        weibull_pdf = weibull_min.pdf(ws_centers, c=k[i], scale=A[i])
        weibull_pdf /= np.sum(weibull_pdf)
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.where(weibull_pdf > 0, hist_scada / weibull_pdf, 0)
        corrections[i, :] = corr

    correction_xr = xr.DataArray(data=corrections, dims=["wd", "ws"], coords={"wd": wd, "ws": ws_centers})
    sim_res = wfm(x, y, h=None, type=0, wd=wd, ws=ws_centers)
    aep_pywake_st = sim_res.aep()
    aep_corrected = aep_pywake_st.transpose('wd', 'ws', 'wt') * correction_xr
    aep_corrected_per_turbine = aep_corrected.sum(dim=['wd','ws'])
    aep_corrected_total_farm = aep_corrected_per_turbine.sum() * 1e3

    aep_pywake = float(aep_pywake_st.isel(wt=np.arange(len(x))).sum()) * 1e3
    aep_pywake_adj = float(aep_corrected_total_farm)

    aep_tot_results.append({
        'year': year,
        'aep_real': f"{aep_real:.2f}",
        'aep_pywake': f"{aep_pywake:.2f}",
        'aep_pywake_adj': f"{aep_pywake_adj:.2f}",
        'error': f"{aep_pywake_adj - aep_real:.2f}",
        'relative error': f"{abs(aep_pywake_adj - aep_real)/aep_real*100:.2f}",
        'overestimated': str(aep_pywake_adj > aep_real)
    })
    
    aep_real_tot += aep_real
    aep_pywake_tot += aep_pywake
    aep_pywake_adj_tot += aep_pywake_adj
    
    for i in range(len(wd)):
        mask = df[df['wind_dir_bin']==wd[i]]
        aep_real=mask['energy'].sum()*1e-6/output_frac
        aep_pywake_i = float(aep_pywake_st.isel(wd=i).sum())*1e3
        aep_pywake_adj=float(aep_corrected.isel(wd=i).sum())*1e3
        aep_tot_results_wd.append({
            'year': year,
            'wind_dir': wd[i],
            'prob':round(f[i]*100,2),
            'A': round(A[i],2),
            'k': round(k[i],2),
            'aep_real': aep_real,
            'aep_pywake': aep_pywake_i,
            'aep_pywake_adj': aep_pywake_adj,
            'error':aep_pywake_adj-aep_real,
            'relative error':abs(aep_pywake_adj-aep_real)/aep_real*100,
            'overestimated': str(aep_pywake_adj>aep_real)
            })


# Append total row
aep_tot_results.append({
    'year': 'Total',
    'aep_real': f"{aep_real_tot:.2f}",
    'aep_pywake': f"{aep_pywake_tot:.2f}",
    'aep_pywake_adj': f"{aep_pywake_adj_tot:.2f}",
    'error': f"{aep_pywake_adj_tot - aep_real_tot:.2f}",
    'relative error': f"{abs(aep_pywake_adj_tot - aep_real_tot)/aep_real_tot*100:.2f}",
    'overestimated': str(aep_pywake_adj_tot > aep_real_tot)
})

# Save results
pd.DataFrame(aep_tot_results).to_csv(f'{save_folder}/aep_results_{n}winddirections_year_corrected.csv', index=False)

#%%

aep_tot_results_wd.append({
    'year': 'Total',
    'wind_dir': 'Total',
    'aep_real' : aep_real_tot,
    'aep_pywake' : aep_pywake_tot,
    'aep_pywake_adj': aep_pywake_adj_tot,
    'error':aep_pywake_adj_tot-aep_real_tot,
    'relative error':abs(aep_pywake_adj_tot-aep_real_tot)/aep_real_tot*100,
    'overestimated': str(aep_pywake_adj_tot>aep_real_tot)
    })
        
            # Convert results to a DataFrame for easier analysis
aep_tot_results_wd_df = pd.DataFrame(aep_tot_results_wd)
aep_tot_results_wd_df.to_csv(f'{save_folder}/wind_aep_results_{n}winddirections.csv')