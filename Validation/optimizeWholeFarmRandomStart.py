# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 14:41:23 2025

@author: erica
"""

import numpy as np
import os
from pathlib import Path
import time
import pandas as pd
from datetime import datetime
from topfarm import TopFarmProblem
from topfarm.cost_models.cost_model_wrappers import CostModelComponent
from topfarm.constraint_components.spacing import SpacingConstraint
from topfarm.constraint_components.boundary import XYBoundaryConstraint, InclusionZone
from topfarm.easy_drivers import EasyScipyOptimizeDriver
from py_wake.wind_farm_models import PropagateUpDownIterative
from py_wake.site import XRSite
from py_wake.wind_turbines import WindTurbine
from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
from py_wake.deficit_models import Rathmann, ZongGaussianDeficit
from py_wake.rotor_avg_models import GaussianOverlapAvgModel
from py_wake.turbulence_models import STF2017TurbulenceModel
import xarray as xr

from support_functions.CustomXYPlotComp import CustomXYPlotComp

# Load Weibull data
load_folder = 'D:/Thesis/Data/Penmanshiel/2025-06-01/'
season_txt = 'no_season'
n = 36
bin_labels = np.arange(0.0, 360.0, 360/n)
weibull_results = pd.read_csv(f'{load_folder}/Weibull/{n}_winddirections/{season_txt}/weibull_results_tot.csv')
f = np.array(weibull_results['wind_dir_probability'])
f[-1] = 1-sum(f[0:-1])
A = np.array(weibull_results['A_mle'])
k = np.array(weibull_results['k_mle'])
ti = np.array(weibull_results['TI'])
u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25.1, 30]
power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 0, 0]
ct = [0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.30, 0.24, 0.19, 0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0, 0]
u_step = 0.1
ws = np.arange(u[0], u[-2], u_step)
wd = bin_labels

# Wind turbine setup
my_wt = WindTurbine(name='my_wt', diameter=82, hub_height=90, powerCtFunction=PowerCtTabular(u, power, 'W', ct))
n_wt = 14

# Site and model setup
site = XRSite(ds=xr.Dataset(data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ('wd', ti)}, coords={'wd': wd}))
site.default_wd = wd
site.default_ws = ws
wfm = PropagateUpDownIterative(site, my_wt, ZongGaussianDeficit(use_effective_ws=True, rotorAvgModel=GaussianOverlapAvgModel()), blockage_deficitModel=Rathmann(use_effective_ws=True), turbulenceModel=STF2017TurbulenceModel())

def aep_func(x, y, **kwargs):
     sim_res = wfm(x, y,   # wind turbine positions
                   # wind turbine heights (defaults to the heights defined in windTurbines)
                   h=None,
                   type=0,   # Wind turbine types
                   wd=wd,    # Wind direction
                   ws=np.arange(u[0], u[-2], u_step))    # Wind speed
     aep_pywake_st = sim_res.aep()
     aep_pywake = float(aep_pywake_st.isel(wt=np.arange(len(x))).sum())
     return aep_pywake

x0 = np.array([164.31855339, 483.54455926, 394.90183711, 664.38319266, 938.66446991, 1194.93045332, 730.02368931, 1073.3739787, 1336.49699347, 1661.5203035, 1124.49003472, 1442.34463091, 1706.46503031, 1975.82170693]) - 164.31855339
y0 = np.array([32.04702276, -245.27312429, 414.66876535, 120.11340467, -140.41630846, -386.15709635, 633.94516069, 308.70000026, 90.98033389, -151.4246062, 721.56676289, 535.09287091, 315.37169586, 27.71042063]) - 32.0470227
boundary = np.array([(0.0, 0.0), (759.3, -838.5), (1404.5, -369.7), (2015.1, -32.4), (2152.5, 105.3), (1030.2, 1024.7), (388.8, 960.5), (112.6, 830.3)]) - np.array([164.31855339, 32.0470227])

plot_comp = CustomXYPlotComp(xlim=(-100, 2500),
                             ylim=(-1100, 1100),
                             xlabel="x [m]",
                             ylabel="y [m]",
                             xticks=[0, 500, 1000, 1500, 2000, 2500],
                             yticks=[-1000, -500, 0, 500, 1000])

save_folder = load_folder + f'Optimization/Whole_farm/{n}winddirections' + 'MultiStart_' + datetime.now().strftime('%Y-%m-%d') + '/'
os.makedirs(save_folder, exist_ok=True)

figure_folder = f"D:/Thesis/Figures/Penmanshiel/Optimization/Whole_farm/{n}winddirections/"+ 'MultiStart_' + datetime.now().strftime('%Y-%m-%d') + "/"
os.makedirs(figure_folder, exist_ok=True)

N_RUNS = 5
np.random.seed(42)
for i in range(N_RUNS):
    x_start = x0 + np.random.uniform(-100, 100, size=len(x0))
    y_start = y0 + np.random.uniform(-100, 100, size=len(y0))

    aep_comp = CostModelComponent(input_keys=['x', 'y'], n_wt=n_wt, cost_function=aep_func, objective=True, maximize=True)
    min_spacing = 2 * my_wt.diameter()
    xyboundconst = [InclusionZone(boundary)]
    constraints = [XYBoundaryConstraint(
        xyboundconst, boundary_type='multi_polygon'), SpacingConstraint(min_spacing)]
    driver = EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=400, tol=1e-4)

    tf = TopFarmProblem(
        design_vars={"x": x_start, "y": y_start},
        cost_comp=aep_comp,
        constraints=constraints,
        driver=driver,
        plot_comp=plot_comp,
        n_wt=n_wt
    )

    t0 = time.time()
    cost, state, recorder = tf.optimize()
    t1 = time.time()
    elapsed = t1 - t0

    rec_dict = tf.get_vars_from_recorder()
    aep = -cost * 1e3
    aep0 = -rec_dict['c'][0] * 1e3

    results = pd.DataFrame([
        {'Run': i, 'AEP': aep, 'Relative increase (%)': (aep - aep0)/aep0*100, 'Iterations': recorder.num_cases, 'Execution time (s)': elapsed},
        {'Run': i, 'AEP': aep0, 'Relative increase (%)': 0, 'Iterations': 0, 'Execution time (s)': 0}
    ])

    run_folder = os.path.join(save_folder, f'run_{i}')
    os.makedirs(run_folder, exist_ok=True)
    results.to_csv(os.path.join(run_folder, 'summary.csv'), index=False)

    # Save layout history
    data = []
    for it in range(len(rec_dict['x'])):
        for wt in range(n_wt):
            data.append({'Iteration': it, 'Turbine': wt, 'x': rec_dict['x'][it][wt], 'y': rec_dict['y'][it][wt], 'AEP': -rec_dict['c'][it] * 1e3})
    pd.DataFrame(data).to_csv(os.path.join(run_folder, 'layout_history.csv'), index=False)
    
    plot_comp.save_fig(figure_folder + f"run_{i}.png")
    
