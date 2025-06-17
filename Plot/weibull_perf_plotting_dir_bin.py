# -*- coding: utf-8 -*-
"""
Created on Sun May 25 19:00:53 2025

@author: erica
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import glob
import matplotlib.ticker as mtick
plt.rcParams.update({'font.size': 22})
#%% Wihtout probability
# Define input and output paths
base_input_folder = "D:/Thesis/Data/Penmanshiel/2025-05-30/Weibull/"
today = datetime.now().strftime('%Y-%m-%d') 
output_folder = f"D:/Thesis/Figures/Penmanshiel/Weibull_perf/{today}/by_direction_bin"
os.makedirs(output_folder, exist_ok=True)

# Containers
yearly_data = {}
seasonal_data = []

# Look into each {n}_winddirections folder
wind_dir_folders = glob.glob(os.path.join(base_input_folder, "*_winddirections"))

for wind_dir_folder in wind_dir_folders:
    n_dirs = os.path.basename(wind_dir_folder).split("_")[0]

    # Yearly + total data (no_season)
    no_season_folder = os.path.join(wind_dir_folder, "no_season")
    if os.path.exists(no_season_folder):
        for file in glob.glob(os.path.join(no_season_folder, "weibull_results_*.csv")):
            df = pd.read_csv(file)
            if 'Wind direction' in df.columns:
                df['wind_dir'] = pd.to_numeric(df['Wind direction'], errors='coerce')
                df['rmse'] = pd.to_numeric(df['RMSE'], errors='coerce')
                df['ks'] = pd.to_numeric(df['K-S'], errors='coerce')
                label = os.path.splitext(os.path.basename(file))[0].replace("weibull_results_", "")
                df['year'] = label
                yearly_data.setdefault(n_dirs, []).append(df[['wind_dir', 'rmse', 'ks', 'year']])

    # Seasonal data
    season_folder = os.path.join(wind_dir_folder, "season")
    if os.path.exists(season_folder):
        for file in glob.glob(os.path.join(season_folder, "weibull_results_*.csv")):
            df = pd.read_csv(file)
            if 'Wind direction' in df.columns and 'season' in df.columns:
                df['wind_dir'] = pd.to_numeric(df['Wind direction'], errors='coerce')
                df['rmse'] = pd.to_numeric(df['RMSE'], errors='coerce')
                df['ks'] = pd.to_numeric(df['K-S'], errors='coerce')
                df['n_dirs'] = n_dirs
                seasonal_data.append(df[['wind_dir', 'rmse', 'ks', 'season', 'n_dirs']])

# --- YEARLY PLOTS (per direction bin) ---
for n_dirs, dfs in yearly_data.items():
    combined = pd.concat(dfs, ignore_index=True)
    for metric, ylabel in [('rmse', 'RMSE'), ('ks', 'K-S Statistic')]:
        plt.figure(figsize=(10, 6))
        if (combined['year']=='tot').all():
            sns.scatterplot(data=combined, x='wind_dir', y=metric, color='tab:blue', legend=False)
        else:
            sns.scatterplot(data=combined, x='wind_dir', y=metric, hue='year', palette='tab10')
            plt.legend(loc='upper right')
        plt.xlabel('Wind Direction Bin (°)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Wind Direction Bin (Yearly, {n_dirs} directions)')
        plt.grid(True)
        plt.tight_layout()
        save_path = os.path.join(output_folder, f"{metric}_vs_dirbin_yearly_{n_dirs}wd.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

# --- SEASONAL PLOTS (per direction bin) ---
if seasonal_data:
    combined_seasonal = pd.concat(seasonal_data, ignore_index=True)
    for n_dirs in combined_seasonal['n_dirs'].unique():
        df_subset = combined_seasonal[combined_seasonal['n_dirs'] == n_dirs]
        for metric, ylabel in [('rmse', 'RMSE'), ('ks', 'K-S Statistic')]:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_subset, x='wind_dir', y=metric,
                            hue='season', palette='tab10')
            plt.xlabel('Wind Direction Bin (°)')
            plt.ylabel(ylabel)
            plt.title(f'{ylabel} vs Wind Direction Bin (Seasonal, {n_dirs} directions)')
            plt.grid(True)
            plt.legend(loc='upper right')
            plt.tight_layout()
            save_path = os.path.join(output_folder, f"{metric}_vs_dirbin_seasonal_{n_dirs}wd.png")
            plt.savefig(save_path, dpi=300)
            plt.close()


#%% With probability
# Define input and output paths
base_input_folder = "D:/Thesis/Data/Penmanshiel/2025-06-01/Weibull/"
today = datetime.now().strftime('%Y-%m-%d') 
output_folder = f"D:/Thesis/Figures/Penmanshiel/Weibull_perf/{today}/by_direction_bin/with_prob/1junedata/"
os.makedirs(output_folder, exist_ok=True)

# Containers
yearly_data = {}
seasonal_data = []

# Look into each {n}_winddirections folder
wind_dir_folders = glob.glob(os.path.join(base_input_folder, "*_winddirections"))

for wind_dir_folder in wind_dir_folders:
    n_dirs = os.path.basename(wind_dir_folder).split("_")[0]

    # Yearly + total data (no_season)
    no_season_folder = os.path.join(wind_dir_folder, "no_season")
    if os.path.exists(no_season_folder):
        for file in glob.glob(os.path.join(no_season_folder, "weibull_results_*.csv")):
            df = pd.read_csv(file)
            if 'Wind direction' in df.columns:
                df['wind_dir'] = pd.to_numeric(df['Wind direction'], errors='coerce')
                df['rmse'] = pd.to_numeric(df['RMSE'], errors='coerce')
                df['ks'] = pd.to_numeric(df['K-S'], errors='coerce')
                df['prob'] = df['Probability'] / 100
                label = os.path.splitext(os.path.basename(file))[0].replace("weibull_results_", "")
                df['year'] = str(label)
                yearly_data.setdefault(n_dirs, []).append(df[['wind_dir', 'prob', 'rmse', 'ks', 'year']])

    # Seasonal data
    season_folder = os.path.join(wind_dir_folder, "season")
    if os.path.exists(season_folder):
        for file in glob.glob(os.path.join(season_folder, "weibull_results_*.csv")):
            df = pd.read_csv(file)
            if 'Wind direction' in df.columns and 'season' in df.columns:
                df['wind_dir'] = pd.to_numeric(df['Wind direction'], errors='coerce')
                df['rmse'] = pd.to_numeric(df['RMSE'], errors='coerce')
                df['ks'] = pd.to_numeric(df['K-S'], errors='coerce')
                df['prob'] = df['Probability'] / 100
                df['n_dirs'] = n_dirs
                seasonal_data.append(df[['wind_dir', 'prob', 'rmse', 'ks', 'season', 'n_dirs']])

# --- YEARLY PLOTS (per direction bin) ---
for n_dirs, dfs in yearly_data.items():
    combined = pd.concat(dfs, ignore_index=True)
    for metric, ylabel in [('rmse', 'RMSE'), ('ks', 'K-S Statistic')]:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        if (combined['year']=='tot').all():
            sns.lineplot(data=combined, x='wind_dir', y=metric, hue='year', style='year',
             legend=False, ax=ax1)
            sns.scatterplot(data=combined, x='wind_dir', y=metric, color='tab:blue', legend=False,ax=ax1)
            maxperc=0.1
        else:
            sns.lineplot(data=combined, x='wind_dir', y=metric, hue='year', style='year',
              legend=False, ax=ax1)
            sns.scatterplot(data=combined, x='wind_dir', y=metric, hue='year', palette='tab10',ax=ax1)
            maxperc=0.25
            ax1.legend(loc='upper right',fontsize=20)
        ax1.set_xlabel('Wind Direction Bin (°)')
        ax1.set_ylabel(ylabel, color='black')
        ax1.tick_params(axis='y', labelcolor='black')
        
        # Right y-axis: Probability
        ax2 = ax1.twinx()
        
        sns.lineplot(data=combined, x='wind_dir', y='prob', hue='year',
                     marker=None, palette='tab10', legend=False, ax=ax2,dashes=True, alpha=0.4)
        ax2.set_ylabel('Probability (%)', color='grey')
        ax2.tick_params(axis='y', labelcolor='grey')
        ax2.set_ylim(0.0, maxperc)  # 1% to 100%
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
        #plt.title(f'{ylabel} vs Wind Direction Bin with Probability Overlay ({n_dirs} directions)')
        
        fig.tight_layout()
        save_path = os.path.join(output_folder, f"{metric}_with_prob_vs_dirbin_yearly_{n_dirs}wd.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

# --- SEASONAL PLOTS (per direction bin) ---
if seasonal_data:
    combined_seasonal = pd.concat(seasonal_data, ignore_index=True)
    for n_dirs in combined_seasonal['n_dirs'].unique():
        df_subset = combined_seasonal[combined_seasonal['n_dirs'] == n_dirs]
        for metric, ylabel in [('rmse', 'RMSE'), ('ks', 'K-S Statistic')]:
            fig, ax1 = plt.subplots(figsize=(10, 6))
            
            # Left y-axis: RMSE or K-S
            lineplot = sns.lineplot(data=df_subset, x='wind_dir', y=metric, hue='season',
                        style='season', linestyle='--', ax=ax1)
            sns.scatterplot(data=df_subset, x='wind_dir', y=metric, hue='season',
                            palette='tab10', legend=False, ax=ax1)
            ax1.set_xlabel('Wind Direction Bin (°)')
            ax1.set_ylabel(ylabel, color='black')
            ax1.tick_params(axis='y', labelcolor='black')
            
            # Right y-axis: Probability
            ax2 = ax1.twinx()
            sns.lineplot(data=df_subset, x='wind_dir', y='prob', hue='season',
                         dashes=False, alpha=0.4,marker=None, palette='tab10', legend=False, ax=ax2)
            ax2.set_ylabel('Probability (%)', color='grey')
            ax2.tick_params(axis='y', labelcolor='grey')
            ax2.set_ylim(0.0, 0.3)  # 1% to 100%
            ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
            #} vs Wind Direction Bin with Probability Overlay ({n_dirs} directions)')
            ax1.legend(loc='upper right',fontsize=20) 
            fig.tight_layout()
            save_path = os.path.join(output_folder, f"{metric}_with_prob_vs_dirbin_season_{n_dirs}wd.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

