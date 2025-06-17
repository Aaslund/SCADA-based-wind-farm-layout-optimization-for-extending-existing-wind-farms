# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:57:44 2025

@author: erica
"""

if __name__ == '__main__':
    # Get the parent directory (A) and add it to sys.path
    from topfarm.cost_models.cost_model_wrappers import CostModelComponent
    from py_wake.utils.gradients import autograd
    from topfarm.plotting import XYPlotComp
    from topfarm.constraint_components.spacing import SpacingConstraint
    from topfarm.easy_drivers import EasyScipyOptimizeDriver
    from topfarm.constraint_components.boundary import XYBoundaryConstraint, CircleBoundaryConstraint, InclusionZone, ExclusionZone
    from topfarm import TopFarmProblem
    import xarray as xr
    from py_wake.site import XRSite
    from py_wake.wind_farm_models import PropagateUpDownIterative
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
    from py_wake.wind_turbines import WindTurbine
    from py_wake.turbulence_models import STF2017TurbulenceModel
    from py_wake.rotor_avg_models import GaussianOverlapAvgModel
    from py_wake.deficit_models import Rathmann
    from py_wake.deficit_models import ZongGaussianDeficit
    from support_functions.create_turbine_rings import create_turbine_rings
    import pandas as pd
    import numpy as np
    import os
    import time
    import matplotlib.pyplot as plt
    from datetime import datetime
    from pathlib import Path
    import sys
    parent_dir = str(Path(__file__).resolve().parent.parent)
    sys.path.append(parent_dir)
    
    from support_functions.CustomXYPlotComp import CustomXYPlotComp
    
    # %%
    n = 12
    wake_angle = 10
    ang_interval = 360/n
    # Define custom bin edges centered around the desired intervals
    # -15 to 375 ensures proper binning
    bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)
    bin_labels = np.arange(0.0, 360.0, ang_interval)
    
    # wind direction
    wd = bin_labels
    
    # %%
    # Create folder path (in current directory)
    load_folder = 'D:/Thesis/Data/Penmanshiel/2025-05-27/'
    season_txt = 'season'
    
    save_folder = load_folder+f'Optimization/Whole_farm/{season_txt}/'
    os.makedirs(save_folder, exist_ok=True)
    
    figure_folder = f"D:/Thesis/Figures/Penmanshiel/Optimization/Whole_farm/{season_txt}/{datetime.now().strftime('%Y-%m-%d')}/"
    os.makedirs(figure_folder, exist_ok=True)
    
    # %%
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
    n_wt = 14
    u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
         16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25.1, 30]
    power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040,
             2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 0, 0]
    ct = [0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.30,
          0.24, 0.19, 0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0, 0]
    
    my_wt = WindTurbine(name='my_wt', diameter=diam,
                        hub_height=hub_height,
                        # ,method='pchip'))
                        powerCtFunction=PowerCtTabular(u, power, 'W', ct))
    u_step = 0.1
    ws = np.arange(u[0], u[-2], u_step)
    
    #%%
    plot_comp = CustomXYPlotComp(xlim=(-100, 2600),
                                 ylim=(-1100, 1100),
                                 xlabel="x [m]",
                                 ylabel="y [m]",
                                 xticks=[0, 500, 1000, 1500, 2000, 2500],
                                 yticks=[-1000, -500, 0, 500, 1000],
                                 existing_x=x,existing_y=y)
    
    min_spacing = 2 * my_wt.diameter()
    circleboundary = create_turbine_rings(x, y, min_spacing/2)
    xyboundconst = [InclusionZone(boundary)]
    for i in range(len(circleboundary)):
        xyboundconst.append(ExclusionZone(circleboundary[i]))
    constraints = [XYBoundaryConstraint(
        xyboundconst, boundary_type='multi_polygon'), SpacingConstraint(min_spacing)]
    
    driver = EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=100, tol=1e-4)
    
    # Load full Weibull seasonal-directional data
    weibull_df = pd.read_csv(f'{load_folder}/Weibull/{n}_winddirections/{season_txt}/weibull_results_tot.csv')
    
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    results = {}
    
    for season in seasons:
        print(f"\n--- Optimizing for {season} ---")
        
        # Filter data for this season
        wdata = weibull_df[weibull_df['season'] == season]
        f = np.array(wdata['wind_dir_probability'])
        A = np.array(wdata['A_mle'])
        k = np.array(wdata['k_mle'])
        
        # Normalize last probability if needed
        f[-1] = 1 - np.sum(f[:-1])
        ti=np.array(wdata['TI'])
        
        # Create seasonal site
        site = XRSite(
            ds=xr.Dataset(data_vars={'Sector_frequency': ('wd', f),
                                     'Weibull_A': ('wd', A),
                                     'Weibull_k': ('wd', k),
                                     'TI': ('wd',ti)},
                          coords={'wd': bin_labels}),
            initial_position=np.array([x, y]).T)
        site.default_wd = wd
        site.default_ws = ws
        
        # Wind farm model
        wfm = PropagateUpDownIterative(
            site, my_wt, ZongGaussianDeficit(use_effective_ws=True, rotorAvgModel=GaussianOverlapAvgModel()),
            blockage_deficitModel=Rathmann(use_effective_ws=True),
            turbulenceModel=STF2017TurbulenceModel()
        )
        
        # Define cost function
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
        
        aep_comp = CostModelComponent(
            input_keys=['x', 'y'],
            n_wt=n_wt,
            cost_function=aep_func,
            objective=True,
            maximize=True
        )
        
        # Setup and run optimization
        tf = TopFarmProblem(
            design_vars={"x": x, "y": y},
            cost_comp=aep_comp,
            constraints=constraints,
            driver=driver,
            plot_comp=plot_comp,
            n_wt=n_wt
        )
        
        start_time = time.perf_counter()
        cost, state, recorder = tf.optimize()
        end_time = time.perf_counter()
        rec_dict = tf.get_vars_from_recorder()
        execution_time = end_time - start_time
        plot_comp.save_fig(figure_folder + f"optimization_{n}winddirections_{season}_iter{recorder.num_cases}.png")
        #%%
        
        aep=-cost
        aep0=-rec_dict['c'][0]
        results=pd.DataFrame({
            'AEP': aep,
            'Relative increase': ((aep - aep0) / aep0 * 100), 
            'Iterations': recorder.num_cases,
            "x": state['x'],
            "y": state['y'],
            "execution_time": execution_time})
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(f'{save_folder}/optimization_{n}winddirections_{season}_iter{recorder.num_cases}.csv', index=False)

        num_iterations = len(rec_dict['x'])
        num_turbines = len(rec_dict['x'][0])  # Assuming all iterations have the same number of turbines
        
        # Prepare long-format data
        data = []
        for i in range(num_iterations):
            for t in range(num_turbines):
                data.append({
                    'Season': season,
                    'Iteration': i,
                    'Turbine': t,
                    'x': rec_dict['x'][i][t],
                    'y': rec_dict['y'][i][t],
                    'AEP': -rec_dict['c'][i]  # Negate if your cost is -AEP
                })

        # Convert to DataFrame
        turbine_positions_df = pd.DataFrame(data)
        turbine_positions_df.to_csv(f'{save_folder}/turbine_positions_{n}winddirections_{season}_iter{recorder.num_cases}.csv', index=False)
