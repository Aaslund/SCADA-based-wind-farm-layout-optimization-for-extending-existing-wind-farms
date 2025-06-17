# -*- coding: utf-8 -*-
"""
Created on Mon May 26 19:04:59 2025

@author: erica
"""

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
    from pathlib import Path
    import sys
    parent_dir = str(Path(__file__).resolve().parent.parent)
    sys.path.append(parent_dir)
    
    class CustomXYPlotComp(XYPlotComp):
        def __init__(self, xlim=None, ylim=None, xlabel='x [m]', ylabel='y [m]', xticks=None, yticks=None, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._custom_xlim = xlim
            self._custom_ylim = ylim
            self._xlabel = xlabel
            self._ylabel = ylabel
            self._xticks = xticks
            self._yticks = yticks
    
        def set_title(self, cost0, cost):
            rec = self.problem.recorder
            iteration = rec.num_cases
            # Get the increase/expansion factor if it exists, otherwise default to 1
            inc_or_exp = getattr(self.problem.cost_comp, 'inc_or_exp', 1.0)
    
            # Compute percentage delta safely
            delta = ((cost - cost0) / cost0 * 100) if cost0 else 0
    
            # Create a clean and informative title
            title = f"Iterations: {iteration} | AEP: {cost * inc_or_exp:.2f} GWh  |  Δ: {delta:+.1f}%"
    
            # Set the title with custom formatting
            self.ax.set_title(title, fontsize=12)
    
        def init_plot(self, limits):
            self.ax.cla()
            self.ax.axis('equal')
    
            # Use custom limits if provided, else fallback to auto
            if self._custom_xlim and self._custom_ylim:
                self.ax.set_xlim(*self._custom_xlim)
                self.ax.set_ylim(*self._custom_ylim)
            else:
                mi = limits.min(0)
                ma = limits.max(0)
                ra = ma - mi + 1
                ext = .1
                xlim, ylim = np.array([mi - ext * ra, ma + ext * ra]).T
                self.ax.set_xlim(xlim)
                self.ax.set_ylim(ylim)
            # Set axis labels
            self.ax.set_xlabel(self._xlabel)
            self.ax.set_ylabel(self._ylabel)
    
            # Set custom ticks if provided
            if self._xticks is not None:
                self.ax.set_xticks(self._xticks)
            if self._yticks is not None:
                self.ax.set_yticks(self._yticks)
    
        def compute(self, inputs, outputs):
            super().compute(inputs, outputs)  # run original logic
            # Then adjust legend
            handles, labels = self.ax.get_legend_handles_labels()
            self.ax.legend(handles, labels, loc='lower right', fontsize=10)
            
    def seasonal_weighted_aep_func(x, y, **kwargs):
        total_aep = 0
        for season, wfm in seasonal_wfms.items():
            sim_res = wfm(x, y, wd=bin_labels, ws=ws)
            seasonal_aep = float(sim_res.aep().isel(wt=np.arange(len(x))).sum())
            total_aep += season_weights[season] * seasonal_aep
        return total_aep
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
    save_folder = load_folder
    os.makedirs(save_folder, exist_ok=True)
    
    # %%
    x = np.array([164.31855339,  483.54455926,  394.90183711,  664.38319266,
                  938.66446991, 1194.93045332,  730.02368931, 1073.3739787,
                  1336.49699347, 1661.5203035, 1124.49003472, 1442.34463091,
                  1706.46503031, 1975.82170693])
    y = np.array([32.04702276, -245.27312429,  414.66876535,  120.11340467,
                  -140.41630846, -386.15709635,  633.94516069,  308.70000026,
                  90.98033389, -151.4246062,  721.56676289,  535.09287091,
                  315.37169586,   27.71042063])
    
    boundary = np.array([
        (0.0, 0.0),
        (759.3000118067339, -838.5008776575139),
        (1404.4887297180828, -369.7348857613972),
        (2015.077636388391, -32.399901569969394),
        (2152.4804284290312, 105.32498167009027),
        (1030.2364721134584, 1024.6572692781397),
        (388.8327571837189, 960.461527003533),
        (112.57349730534965, 830.3425848119384)
    ])
    
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
                        powerCtFunction=PowerCtTabular(u, power, 'kW', ct))
    u_step = 0.1
    ws = np.arange(u[0], u[-2], u_step)
    plot_comp = CustomXYPlotComp(xlim=(-100, 2700),
                                 ylim=(-1000, 1100),
                                 xlabel="x [m]",
                                 ylabel="y [m]",
                                 xticks=[0, 500, 1000, 1500, 2000],
                                 yticks=[-1000, -500, 0, 500, 1000])
    
    
    min_spacing = 2 * my_wt.diameter()
    circleboundary = create_turbine_rings(x, y, min_spacing/2)
    xyboundconst = [InclusionZone(boundary)]
    for i in range(len(circleboundary)):
        xyboundconst.append(ExclusionZone(circleboundary[i]))
    constraints = [XYBoundaryConstraint(
        xyboundconst, boundary_type='multi_polygon'), SpacingConstraint(min_spacing)]
    
    driver = EasyScipyOptimizeDriver(optimizer='COBYLA', maxiter=1000, tol=1e-6)
    ec = 10
    
    # Load full Weibull seasonal-directional data
    weibull_df = pd.read_csv(f'{load_folder}/Weibull/{n}_winddirections/{season_txt}/weibull_results_tot.csv')
    
    seasons = ['Winter', 'Spring', 'Summer', 'Autumn']
    season_weights = {'Winter': 0.25, 'Spring': 0.25, 'Summer': 0.25, 'Autumn': 0.25}  # or adjust
    seasonal_wfms = {}
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
                                     'TI': ('wd', ti)},
                          coords={'wd': wd}),
            initial_position=np.array([x, y]).T)
        site.default_wd = wd
        site.default_ws = ws
        
        # Wind farm model
        wfm = PropagateUpDownIterative(
            site, my_wt, ZongGaussianDeficit(use_effective_ws=True, rotorAvgModel=GaussianOverlapAvgModel()),
            blockage_deficitModel=Rathmann(use_effective_ws=True),
            turbulenceModel=STF2017TurbulenceModel()
        )
        seasonal_wfms[season] = wfm
        
    aep_comp = CostModelComponent(
        input_keys=['x', 'y'],
        n_wt=n_wt,
        cost_function=seasonal_weighted_aep_func,
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
        expected_cost=ec,
        n_wt=n_wt
    )
        
    start_time = time.perf_counter()
    cost, state, recorder = tf.optimize()
    end_time = time.perf_counter()
    
    exec_time = end_time - start_time
    print(f"AEP: {cost:.2f} MWh | Time: {exec_time:.1f} s")
    
    
    # Optionally: save or plot results
