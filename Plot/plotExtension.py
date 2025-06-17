# -*- coding: utf-8 -*-
"""
Created on Tue May 27 10:40:09 2025

@author: erica
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 24 13:05:47 2025

@author: erica
"""

if __name__ == '__main__':
# AEP validation 3

    import pandas as pd
    import numpy as np
    import os
    import matplotlib.pyplot as plt
    from pathlib import Path
    import sys
    # Get the parent directory (A) and add it to sys.path
    parent_dir = str(Path(__file__).resolve().parent.parent)
    sys.path.append(parent_dir)
    
    from datetime import datetime
    #from support_functions.ct_eval import ct_eval
    
    from py_wake.deficit_models import ZongGaussianDeficit
    from py_wake.deficit_models import Rathmann
    from py_wake.rotor_avg_models import GaussianOverlapAvgModel
    from py_wake.turbulence_models import STF2017TurbulenceModel
    from py_wake.wind_turbines import WindTurbine
    from py_wake.wind_turbines.power_ct_functions import PowerCtTabular
    from py_wake.wind_farm_models import PropagateUpDownIterative
    
    from py_wake.site import XRSite
    import xarray as xr
    from py_wake.flow_map import XYGrid
    from matplotlib import cm
    from py_wake.utils.plotting import setup_plot
    from matplotlib.colors import ListedColormap
    #%%
    
    # Create folder path (in current directory)
    load_folder = 'D:/Thesis/Data/Penmanshiel/2025-05-27/'
    season_txt = 'no_season'
    figure_folder = f"D:/Thesis/Figures/Penmanshiel/Optimization/Extension/Wake/{datetime.now().strftime('%Y-%m-%d')}/"
    os.makedirs(figure_folder, exist_ok=True)
    save_folder = f"D:/Thesis/Figures/Penmanshiel/Wake/{datetime.now().strftime('%Y-%m-%d')}/"
    os.makedirs(save_folder, exist_ok=True)
    #%%
    n = 12
    iterations=96
    ang_interval = 360/n
    # Define custom bin edges centered around the desired intervals
    bin_edges = np.arange(-ang_interval/2, 360+ang_interval/2, ang_interval)  # -15 to 375 ensures proper binning
    bin_labels = np.arange(0.0, 360.0, ang_interval)
    u_step=0.1
    optimization_select=f'{n}winddirections_{season_txt}_iter{iterations}'
    #%%
    wdplot=210
    wsplot=7
    #%%
    x=np.array([   0.        ,  319.22363416,  230.58157057,  500.06092411,
            774.34016385, 1030.60424373,  565.70093313,  909.048672  ,
           1172.16973239, 1497.19062847,  960.16434834, 1278.01658368,
           1542.1350215 , 1811.48969781])
    y=np.array([   0.        , -277.32014705,  382.62174258,   88.0663819 ,
           -172.46333123, -418.20411911,  601.89813793,  276.65297749,
             58.93331112, -183.47162896,  689.51974012,  503.04584814,
            283.32467309,   -4.33660214])
    
    opti_df = pd.read_csv(f'{load_folder}/Optimization/Extension/{optimization_select}.csv')
    x_opti=np.array([opti_df['x1'][0],opti_df['x2'][0]])	
    y_opti=np.array([opti_df['y1'][0],opti_df['y2'][0]])
    xfinal=np.append(x,x_opti)
    yfinal=np.append(y,y_opti)
    
    hub_height = 90
    diam = 82
    n_wt=16
    
    #u = np.concatenate(([0.1], [3.5], np.arange(4, 14.5, 0.5), [25], [25.1], [35]))
    #power = [0, 0.1, 55,110, 186, 264, 342, 424, 506, 618, 730, 865, 999, 1195, 1391,
    #         1558, 1724, 1829, 1909, 1960, 2002, 2025, 2044, 2050,0,0]
    #ct, power_interp, u_highres = ct_eval(u,power)
    u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,25.1,30]
    power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050,0,0]
    ct = [0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.30, 0.24, 0.19, 0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05,0,0]
    my_wt = WindTurbine(name='my_wt',diameter=diam,
                        hub_height=hub_height,
                        powerCtFunction=PowerCtTabular(u,power,'kW',ct))#,method='pchip'))
    
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
        ds=xr.Dataset(data_vars={'Sector_frequency': ('wd', f), 'Weibull_A': ('wd', A), 'Weibull_k': ('wd', k), 'TI': ('wd',ti)},
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
                ws=np.arange(u[0], u[-2], u_step)
                )
    aep_pywake_st=sim_res.aep()
    #%%
    levels=100
    fm=wfm(x=xfinal,y=yfinal,wd=wdplot, ws=wsplot, yaw=0).flow_map(XYGrid(x=np.linspace(min(xfinal)-1.9*diam, max(xfinal)+1.9*diam, 200), y=np.linspace(min(yfinal)-1.9*diam, max(yfinal)+1.9*diam, 200)))
    
    cmap = np.r_[[[1,1,1,1],[1,1,1,1]],cm.Blues(np.linspace(-0,1,128))]
    fig, ax = plt.subplots()
    setup_plot(ax=ax,grid=False, ylabel="y/D [-]", xlabel= "x/D [-]",
               xlim=[fm.x.min()/diam-2, fm.x.max()/diam+2], ylim=[fm.y.min()/diam-2, fm.y.max()/diam+2], axis='auto')
    
    fm.plot(fm.ws - fm.WS_eff, clabel='Deficit [m/s]', levels=levels, cmap=ListedColormap(cmap), normalize_with=diam,ax=ax)
    plt.tight_layout()
    plt.savefig(figure_folder+optimization_select+'_deficit')
    
    fig, ax = plt.subplots()
    setup_plot(ax=ax,grid=False, ylabel="y/D [-]", xlabel= "x/D [-]",
               xlim=[fm.x.min()/diam-2, fm.x.max()/diam+2], ylim=[fm.y.min()/diam-2, fm.y.max()/diam+2], axis='auto')
    contour=fm.plot(fm.WS_eff, clabel='Wind speed [m/s]', levels=levels, cmap='jet', normalize_with=diam,ax=ax)
    #contour.set_clim(6, 6.9)  # <-- force color scale
    #contour.colorbar.update_normal(contour)
    plt.tight_layout()
    plt.savefig(figure_folder+optimization_select+'_velocity')