# -*- coding: utf-8 -*-
"""
Created on Fri May 30 14:06:16 2025

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
    
    from scipy.stats import weibull_min
    
    #from support_functions.ct_eval import ct_eval
    
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
    
    plt.rcParams.update({'font.size': 15})
    
    def circular_mean(degrees):
        radians = np.deg2rad(degrees)
        sin_sum = np.mean(np.sin(radians))
        cos_sum = np.mean(np.cos(radians))
        mean_angle = np.arctan2(sin_sum, cos_sum)
        return np.rad2deg(mean_angle) % 360
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

    season_txt = 'no_season'
    aep_tot_results=[]
    f_list=[]
    df_ws = pd.read_csv(f'{load_folder}df_windspeed_tot.csv')
    df = pd.read_csv(f'{load_folder}df_energy_tot.csv')
    df['date_time'] = pd.to_datetime(df['date_time'])    
    df_ws = df_ws.groupby('date_time').agg({
        'wind_speed': 'mean',
        'wind_dir': circular_mean
        }).reset_index()    
    df_ws['date_time'] = pd.to_datetime(df_ws['date_time'])
    
    start_date = pd.to_datetime('2017-01-01')
    end_date = pd.to_datetime('2020-12-31 23:50:00')
    
    save_folder = load_folder + '/consensus/'
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
    ws_rated=14.5
    #u = np.concatenate(([0.1], [3.5], np.arange(4, 14.5, 0.5), [25], [25.1], [35]))
    #power = [0, 0.1, 55,110, 186, 264, 342, 424, 506, 618, 730, 865, 999, 1195, 1391,
    #         1558, 1724, 1829, 1909, 1960, 2002, 2025, 2044, 2050,0,0]
    #ct, power_interp, u_highres = ct_eval(u,power)
    u = [3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,25.1,30]
    power = [0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050,0,0]
    ct = [0, 0.87, 0.79, 0.79, 0.79, 0.79, 0.78, 0.72, 0.63, 0.51, 0.38, 0.30, 0.24, 0.19, 0.16, 0.13, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05,0,0]
    my_wt = WindTurbine(name='my_wt',diameter=diam,
                        hub_height=hub_height,
                        powerCtFunction=PowerCtTabular(u,power,'W',ct))#,method='pchip'))
    
    #%%
    time_mask_ws = (df_ws['date_time'] >= start_date) & (df_ws['date_time'] <= end_date)
    time_mask = (df['date_time'] >= start_date) & (df['date_time'] <= end_date)
    start_date_data = df['date_time'].min()
    end_date_data = df['date_time'].max()
    if (start_date_data != start_date) |(end_date_data != end_date):
        print("WARNING start or end date not matching between data and script")
    # Apply the mask to your DataFrame
    df_ws = df_ws[time_mask_ws]
    df = df[time_mask]
    tot_year_fraction=((end_date_data-start_date_data).days+1)/365
    
    weibull_results = pd.read_csv(f'{load_folder}/Weibull/{n}_winddirections/consensus/{season_txt}/weibull_results_tot.csv')
    
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
    
    #%% Bin-wise correction
    ws_pywake = aep_pywake_st.ws.values  # exact ws bins used by PyWake
    wd_pywake = aep_pywake_st.wd.values  # exact wd bins
    
    ws_bins = np.append(ws_pywake, ws_pywake[-1] + u_step)  # extend to close final bin
    ws_centers = ws_pywake  # use exact centers from PyWake
    
    df_ws.loc[:,'wind_dir'] = df_ws['wind_dir'].apply(lambda wd: wd - 360 if wd >= 360-ang_interval/2 else wd)
    df_ws['wind_dir_bin'] = pd.cut(df_ws['wind_dir'], bins=bin_edges, labels=bin_labels, right=False).astype(float)
    
    
    df.loc[:,'wind_dir'] = df['wind_dir'].apply(lambda wd: wd - 360 if wd >= 360-ang_interval/2 else wd)
    df['wind_dir_bin'] = pd.cut(df['wind_dir'], bins=bin_edges, labels=bin_labels, right=False).astype(float)
    n_ws = len(ws_centers)
    
    df_hourly = df.resample('h', on='date_time').agg({
        'wind_speed': 'mean',
        'wind_dir': circular_mean,
        }).reset_index()
    df_hourly.loc[:,'wind_dir'] = df_hourly['wind_dir'].apply(lambda wd: wd - 360 if wd >= 360-ang_interval/2 else wd)
    df_hourly['wind_dir_bin'] = pd.cut(df_hourly['wind_dir'], bins=bin_edges, labels=bin_labels, right=False).astype(float)

#%%
    # Storage for correction factors (n_wd, n_ws)
    corrections = np.zeros((n, n_ws))
    # --- Empirical wind speed distribution (from SCADA) ---
    for i, wdi in enumerate(wd):
        df_wdi = df_hourly[df_hourly["wind_dir_bin"] == wdi]
        df_ws_wdi = df_ws[df_ws["wind_dir_bin"] == wdi]
        # Empirical histogram from SCADA
        hist_scada, _ = np.histogram(df_wdi["wind_speed"], bins=ws_bins, density=True)
        hist_scada_ws, _ = np.histogram(df_ws_wdi["wind_speed"], bins=ws_bins, density=True)
    
        #hist_scada /= np.sum(hist_scada)
        # Theoretical Weibull PDF for that direction
        weibull_pdf = weibull_min.pdf(ws_centers, c=k[i], scale=A[i])
        #weibull_pdf /= np.sum(weibull_pdf)  # normalize
    
        # Bin-wise correction - we divide the actual distribution with the modeled one
        # if the match was perfect this would only give 1 every time
        # but we know that the outlier removal removes more large wind speeds than
        # small ones so we will try to multiply down the high wind speeds
        # errstate to avoid dividing by 0
        # if weibull_pdf is above 0 the correction factor for that bin is hist_scada[i]/ weibull_pdf[i]
        # otherwise it is 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            corr = np.where(hist_scada_ws > 0, hist_scada / hist_scada_ws, 0)
            #corr = np.where(ws_centers > 20, 1, corr)
        # we do this for every wind direction
        corrections[i, :] = corr
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ws_centers = 0.5 * (ws_bins[:-1] + ws_bins[1:])
        bin_width=ws_centers[1]-ws_centers[0]
        ax1.bar(ws_centers, hist_scada, width=bin_width, alpha=0.6, label='SCADA Distribution (energy)', color='tab:blue', edgecolor='black', align='center')
        ax1.bar(ws_centers, hist_scada_ws, width=bin_width, alpha=0.6, label='SCADA Distribution (wind speed)', color='tab:orange', edgecolor='black', align='center')
        
        ax1.set_xlabel('Wind Speed [m/s]')
        ax1.set_ylabel('Probability Density [-]', color='k')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.grid(True, linestyle='--', alpha=0.5)
        
        # Create secondary axis (right) for correction factor
        ax2 = ax1.twinx()
        ax2.plot(ws_centers, corr, 'g-', linewidth=2, label='Correction Factor')
        ax2.set_ylabel('Correction Factor [-]', color='g')
        ax2.tick_params(axis='y', labelcolor='g')
        
        # Combine legends from both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        
    # Wrap in xarray for compatibility with PyWake output
    correction_xr = xr.DataArray(
        data=corrections,
        dims=["wd", "ws"],
        coords={"wd": wd_pywake, "ws": ws_pywake},
        name="correction_factors"
        )
    #%% Convert correction to array
    correction = correction_xr.values # shape: (n_wd, n_ws)
    # Reorder aep_pywake_st to (n_wd, n_ws, n_wt) for broadcasting
    aep_reordered = aep_pywake_st.transpose('wd', 'ws', 'wt')  # shape: (n_wd, n_ws, n_wt)
    
    # Apply correction (broadcast over turbines)
    aep_corrected = aep_reordered * correction_xr  # shape: (n_wd, n_ws, n_wt)
    
    # Sum over wind direction and wind speed to get per-turbine AEP
    aep_corrected_per_turbine = aep_corrected.sum(dim=['wd','ws'])  # shape: (n_wt,)
    aep_corrected_per_wd = aep_corrected.sum(dim=['wt','ws'])
    # Optionally, total farm AEP:
    aep_corrected_total_farm = aep_corrected_per_turbine.sum()*1e3
    #%%
    aep_tot_results=[]
    for i in range(1,n_wt+1):
        mask = df[df['turbine_id']==i]
        aep_real=mask['energy'].sum()*1e-6/tot_year_fraction
        aep_pywake_i = float(aep_pywake_st.isel(wt=i-1).sum())*1e3 # MWh
        aep_pywake_adj=float(aep_corrected_per_turbine.isel(wt=i-1))*1e3
        aep_tot_results.append({
            'turbine_id': i,
            'aep_real': f"{aep_real:.2f}",
            'aep_pywake': f"{aep_pywake_i:.2f}",
            'aep_pywake_adj': f"{aep_pywake_adj:.2f}",
            'error':f"{aep_pywake_adj-aep_real:.2f}",
            'relative error':f"{abs(aep_pywake_adj-aep_real)/aep_real*100:.2f}",
            'overestimated': str(aep_pywake_adj>aep_real)
            })
    aep_pywake = float(aep_pywake_st.isel(wt=np.arange(len(x))).sum())*1e3
    aep_pywake_adj = float(aep_corrected_total_farm)
    aep_real = df['energy'].sum()*1e-6/tot_year_fraction
    aep_tot_results.append({
        'turbine_id': 'Total',
        'aep_real' : f"{aep_real:.2f}",
        'aep_pywake' : f"{aep_pywake:.2f}",
        'aep_pywake_adj': f"{aep_pywake_adj:.2f}",
        'error':f"{aep_pywake_adj-aep_real:.2f}",
        'relative error':f"{abs(aep_pywake_adj-aep_real)/aep_real*100:.2f}",
        'overestimated': str(aep_pywake_adj>aep_real)
        })
            
                # Convert results to a DataFrame for easier analysis
    aep_tot_results_df = pd.DataFrame(aep_tot_results)
    aep_tot_results_df.to_csv(f'{save_folder}/aep_results_{n}winddirections_{season_txt}_corr.csv')
    
    #%%
    aep_tot_results_wd=[]
    for i in range(len(wd)):
        mask = df[df['wind_dir_bin']==wd[i]]
        aep_real=mask['energy'].sum()*1e-6/tot_year_fraction
        aep_pywake_i = float(aep_pywake_st.isel(wd=i).sum())*1e3 # MWh
        aep_pywake_adj=float(aep_corrected_per_wd.isel(wd=i))*1e3
        aep_tot_results_wd.append({
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
    aep_pywake = float(aep_pywake_st.isel(wt=np.arange(len(x))).sum())*1e3
    aep_pywake_adj = float(aep_corrected_total_farm)
    aep_real = df['energy'].sum()*1e-6/tot_year_fraction
    aep_tot_results_wd.append({
        'wind_dir': 'Total',
        'aep_real' : aep_real,
        'aep_pywake' : aep_pywake,
        'aep_pywake_adj': aep_pywake_adj,
        'error':aep_pywake_adj-aep_real,
        'relative error':abs(aep_pywake_adj-aep_real)/aep_real*100,
        'overestimated': str(aep_pywake_adj>aep_real)
        })
            
                # Convert results to a DataFrame for easier analysis
    aep_tot_results_wd_df = pd.DataFrame(aep_tot_results_wd)
    aep_tot_results_wd_df.to_csv(f'{save_folder}/wind_aep_results_{n}winddirections_{season_txt}_corr.csv')