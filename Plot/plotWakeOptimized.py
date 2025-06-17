# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 22:04:57 2025

@author: erica
"""

# -*- coding: utf-8 -*-
"""
Modified version of plotWake.py using optimized turbine layout
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from datetime import datetime
from py_wake.deficit_models import ZongGaussianDeficit, Rathmann
from py_wake.rotor_avg_models import GaussianOverlapAvgModel
from py_wake.turbulence_models import STF2017TurbulenceModel
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.wind_farm_models import PropagateUpDownIterative
from py_wake.site import XRSite
from py_wake.site.shear import PowerShear
from py_wake.flow_map import XYGrid
from py_wake.utils.plotting import setup_plot
from py_wake.utils.parallelization import multiprocessing
from matplotlib import cm
from matplotlib.colors import ListedColormap
import xarray as xr



iterations=501
# Setup file structure and load data
load_folder = 'D:/Thesis/Data/Penmanshiel/2025-06-01/'
save_folder = f"D:/Thesis/Figures/Penmanshiel/Wake/WholeFarm/{datetime.now().strftime('%Y-%m-%d')}/"
os.makedirs(save_folder, exist_ok=True)

metadata = pd.read_csv(f'{load_folder}/turbine_year_metadata.csv')
df = pd.read_csv(f'{load_folder}df_energy_tot.csv')
df['date_time'] = pd.to_datetime(df['date_time'])

start_date = pd.to_datetime('2017-01-01')
end_date = pd.to_datetime('2020-12-31 23:50:00')
time_mask = (df['date_time'] >= start_date) & (df['date_time'] <= end_date)
df = df[time_mask]

# Weibull data
season_txt = 'no_season'
n = 12
ang_interval = 360/n
bin_labels = np.arange(0.0, 360.0, ang_interval)
weibull_results = pd.read_csv(f'{load_folder}/Weibull/{n}_winddirections/{season_txt}/weibull_results_tot.csv')

f = np.array(weibull_results['wind_dir_probability'])
f[-1] = 1 - sum(f[0:-1])
A = np.array(weibull_results['A_mle'])
k = np.array(weibull_results['k_mle'])
ti = np.array(weibull_results['TI'])
wd = bin_labels

turbine_positions_path = load_folder + f"Optimization/Whole_farm/turbine_positions_{n}winddirections_no_season_iter{iterations}.csv"
turbine_df = pd.read_csv(turbine_positions_path)
latest_iter = turbine_df["Iteration"].max()
latest_positions = turbine_df[turbine_df["Iteration"] == latest_iter]

optimized_x = latest_positions["x"]
optimized_y = latest_positions["y"]

hub_height = 90
diam = 82
n_wt = len(optimized_x)

# Create site and turbine model
site = XRSite(
    ds=xr.Dataset(data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ('wd', ti)},
                  coords={'wd': wd}), initial_position=np.array([optimized_x, optimized_y]).T)

u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25.1, 30]
power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 0, 0]
ct = [0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.30, 0.24, 0.19, 0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0, 0]
my_wt = WindTurbine(name='my_wt', diameter=diam, hub_height=hub_height, powerCtFunction=PowerCtTabular(u, power, 'kW', ct))

# Setup wind farm model
wdm = ZongGaussianDeficit(use_effective_ws=True, rotorAvgModel=GaussianOverlapAvgModel())
wfm = PropagateUpDownIterative(site, my_wt, wdm,
                                blockage_deficitModel=Rathmann(use_effective_ws=True),
                                turbulenceModel=STF2017TurbulenceModel())

# Simulation
u_step = 0.1
sim_res = wfm(optimized_x, optimized_y, h=None, type=0, wd=wd, ws=np.arange(u[0], u[-2], u_step))
aep_result = sim_res.aep()

# Wake deficit map for 210 deg, 7 m/s
fm = wfm(x=optimized_x, y=optimized_y, wd=210, ws=7, yaw=0).flow_map(
    XYGrid(x=np.linspace(min(optimized_x)-2*diam, max(optimized_x)+2*diam, 200),
           y=np.linspace(min(optimized_y)-2*diam, max(optimized_y)+2*diam, 200)))

cmap = np.r_[[[1,1,1,1],[1,1,1,1]], cm.Blues(np.linspace(-0,1,128))]
fm.plot(fm.ws - fm.WS_eff, clabel='Deficit [m/s]', levels=100,
        cmap=ListedColormap(cmap), normalize_with=1)
setup_plot(grid=False, ylabel="y [m]", xlabel="x [m]",
           xlim=[fm.x.min(), fm.x.max()], ylim=[fm.y.min(), fm.y.max()], axis='auto')

# Save the figure
plt.savefig(f'{save_folder}/wake_deficit_210deg_{n}winddirections_{season_txt}.png', dpi=300)
plt.show()
