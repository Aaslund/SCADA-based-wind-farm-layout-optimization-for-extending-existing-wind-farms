# -*- coding: utf-8 -*-
"""
Created on Mon May 26 11:53:51 2025

@author: erica
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 10:16:36 2025

@author: erica
"""

if __name__ == '__main__':
    import pandas as pd
    import numpy as np
    import os
    import time
    from datetime import datetime
    from pathlib import Path
    import sys
    # Get the parent directory (A) and add it to sys.path
    parent_dir = str(Path(__file__).resolve().parent.parent)
    sys.path.append(parent_dir)
    
    
    from support_functions.create_turbine_rings import create_turbine_rings
    
    from py_wake.deficit_models import ZongGaussianDeficit
    from py_wake.deficit_models import Rathmann
    from py_wake.rotor_avg_models import GaussianOverlapAvgModel
    from py_wake.turbulence_models import STF2017TurbulenceModel
    from py_wake.wind_turbines import WindTurbine
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
    from py_wake.wind_farm_models import PropagateUpDownIterative
    
    from py_wake.site import XRSite
    import xarray as xr
    
    from topfarm import TopFarmProblem
    from topfarm.constraint_components.boundary import XYBoundaryConstraint, CircleBoundaryConstraint, InclusionZone, ExclusionZone
    from topfarm.easy_drivers import EasyScipyOptimizeDriver
    from topfarm.constraint_components.spacing import SpacingConstraint
    from topfarm.plotting import XYPlotComp
    
    from py_wake.utils.gradients import autograd
    from topfarm.cost_models.cost_model_wrappers import CostModelComponent
    from support_functions.CustomXYPlotComp import CustomXYPlotComp
    
    def aep_func(x, y, **kwargs):
        x_stat = np.array([164.31855339,  483.54455926,  394.90183711,  664.38319266,
                      938.66446991, 1194.93045332,  730.02368931, 1073.3739787,
                      1336.49699347, 1661.5203035, 1124.49003472, 1442.34463091,
                      1706.46503031, 1975.82170693])-164.31855339
        y_stat = np.array([32.04702276, -245.27312429,  414.66876535,  120.11340467,
                      -140.41630846, -386.15709635,  633.94516069,  308.70000026,
                      90.98033389, -151.4246062,  721.56676289,  535.09287091,
                      315.37169586,   27.71042063])-32.04702276
        x_all = np.append(x_stat, x)
        y_all = np.append(y_stat, y)
        sim_res = wfm(x_all, y_all,   # wind turbine positions
                      # wind turbine heights (defaults to the heights defined in windTurbines)
                      h=None,
                      type=0,   # Wind turbine types
                      wd=wd,    # Wind direction
                      ws=np.arange(u[0], u[-2], u_step))    # Wind speed
        aep_pywake_st = sim_res.aep()
        aep_pywake = float(aep_pywake_st.isel(wt=np.arange(len(x_all))).sum())
        return aep_pywake
    
    
    #%%
    n = 12
    ang_interval = 360/n
    # Define custom bin edges centered around the desired intervals
    bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
    bin_labels = np.arange(0.0, 360.0, ang_interval)
    
    # wind direction
    wd = bin_labels
    
    #%%
    # Create folder path (in current directory)
    load_folder = 'D:/Thesis/Data/Penmanshiel/2025-06-01/'
    season_txt = 'no_season'
    figure_folder = f"D:/Thesis/Figures/Penmanshiel/Optimization/Extension/{datetime.now().strftime('%Y-%m-%d')}/"
    os.makedirs(figure_folder, exist_ok=True)
    save_folder = load_folder+'Optimization/Extension/'
    os.makedirs(save_folder, exist_ok=True)
    
    #%%
    x = np.array([164.31855339,  483.54455926,  394.90183711,  664.38319266,
                  938.66446991, 1194.93045332,  730.02368931, 1073.3739787,
                  1336.49699347, 1661.5203035, 1124.49003472, 1442.34463091,
                  1706.46503031, 1975.82170693])-164.31855339
    y = np.array([32.04702276, -245.27312429,  414.66876535,  120.11340467,
                  -140.41630846, -386.15709635,  633.94516069,  308.70000026,
                  90.98033389, -151.4246062,  721.56676289,  535.09287091,
                  315.37169586,   27.71042063])-32.04702276
    
    boundary = np.array([
        (0.0, 0.0),
        (759.3000118067339, -838.5008776575139),
        (1404.4887297180828, -369.7348857613972),
        (2015.077636388391, -32.399901569969394),
        (2152.4804284290312, 105.32498167009027),
        (1030.2364721134584, 1024.6572692781397),
        (388.8327571837189, 960.461527003533),
        (112.57349730534965, 830.3425848119384)
    ])- (164.31855339, 32.04702276)

    
    hub_height = 90
    diam = 82
    n_wt=14
    u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,25.1,30]
    power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050,0,0]
    ct = [0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.30, 0.24, 0.19, 0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05,0,0]
    
    my_wt = WindTurbine(name='my_wt',diameter=diam,
                        hub_height=hub_height,
                        powerCtFunction=PowerCtTabular(u,power,'W',ct))#,method='pchip'))
    
    u_step = 0.1
    ws = np.arange(u[0], u[-2], u_step)
    #%%
    plot_comp = CustomXYPlotComp(xlim=(-100, 2600),
                                 ylim=(-1100, 1100),
                                 xlabel="x [m]",
                                 ylabel="y [m]",
                                 xticks=[0, 500, 1000, 1500, 2000, 2500],
                                 yticks=[-1000, -500, 0, 500, 1000],
                                 existing_x=x,existing_y=y,
                                 fontsizelegend=10)
    #%% Movable turbines
    x_mov=np.array([802.77,450])- 164.31855339
    y_mov=np.array([-522.59,750])- 32.04702276
    
    # x_mov=np.array([1000,1400])
    # y_mov=np.array([0,200])
    n_wt_mov=len(x_mov)
    
    x_all = np.append(x, x_mov) 
    y_all = np.append(y, y_mov)
    
    #%%
    weibull_results = pd.read_csv(f'{load_folder}/Weibull/{n}_winddirections/{season_txt}/weibull_results_tot.csv')
    # frequency of the different wind directions
    f = np.array(weibull_results['wind_dir_probability'])
    f[-1]=1-sum(f[0:-1])
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
                      coords={'wd': wd}), initial_position=np.array([x_all, y_all]).T)
    site.default_wd = wd
    site.default_ws = ws
    #%% Make wind deficit model and wind farm model
    wdm = ZongGaussianDeficit(use_effective_ws=True,
                              rotorAvgModel=GaussianOverlapAvgModel()
                              )
    wfm = PropagateUpDownIterative(site, my_wt, wdm,
                                  blockage_deficitModel=Rathmann(use_effective_ws=True),
                                  turbulenceModel=STF2017TurbulenceModel())
    
    #%% Optimization with TopFarm
    aep_comp = CostModelComponent(input_keys=['x','y'], n_wt=n_wt_mov, cost_function=aep_func, objective=True, maximize=True)
    min_spacing = 2 * my_wt.diameter()
    circleboundary = create_turbine_rings(x, y, min_spacing/2)
    xyboundconst=[InclusionZone(boundary)]
    for i in range(len(circleboundary)):
        xyboundconst.append(ExclusionZone(circleboundary[i]))
    constraints=[XYBoundaryConstraint(xyboundconst, boundary_type='multi_polygon'),SpacingConstraint(min_spacing)]
    driver = EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=500, tol=1e-6)
    
    #%%
    tf = TopFarmProblem(
        design_vars={"x": x_mov, "y": y_mov},
        cost_comp=aep_comp,
        constraints=constraints,
        driver=driver,
        plot_comp=plot_comp,
        n_wt=n_wt_mov
    )
    
    start_time = time.perf_counter()
    cost, state, recorder = tf.optimize()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    plot_comp.save_fig(figure_folder + f"optimization_{n}winddirections_{season_txt}_iter{recorder.num_cases}.png")
    #%%
    rec_dict = tf.get_vars_from_recorder()
    aep=-cost*1e3
    aep0=-rec_dict['c'][0]*1e3
    results_summary = {
    'AEP': aep,
    'Relative increase': ((aep - aep0) / aep0 * 100), 
    'Iterations': recorder.num_cases,
    'execution_time': execution_time,
    'x1': state['x'][0],
    'y1': state['y'][0],
    'x2': state['x'][1],
    'y2': state['y'][1]
    }
    rec_history = pd.DataFrame({
        'AEP': -rec_dict['c']*1e3,
        'Iterations': recorder['counter'],
        'x1': rec_dict['x'][:, 0],
        'y1': rec_dict['y'][:, 0],
        'x2': rec_dict['x'][:, 1],
        'y2': rec_dict['y'][:, 1]
    })
    
    # Combine and save both
    summary_df = pd.DataFrame([results_summary])  # single-row summary
    final_df = pd.concat([summary_df, rec_history], axis=0, ignore_index=True)
    final_df.to_csv(f'{save_folder}/{n}winddirections_{season_txt}_iter{recorder.num_cases}.csv', index=False)
    
    