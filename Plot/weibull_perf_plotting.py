import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
import glob
plt.rcParams.update({'font.size': 12})
# Setup
base_input_folder = "D:/Thesis/Data/Penmanshiel/2025-05-24/Weibull/"
output_folder = f"D:/Thesis/Figures/Penmanshiel/Weibull_perf/{datetime.now().strftime('%Y-%m-%d')}"
os.makedirs(output_folder, exist_ok=True)

# Initialize containers for yearly and seasonal data
yearly_data = {}
seasonal_data = []

# Check all subfolders (12_winddirections, 36_winddirections)
wind_dir_folders = glob.glob(os.path.join(base_input_folder, "*_winddirections"))

for wind_dir_folder in wind_dir_folders:
    n_dirs = os.path.basename(wind_dir_folder).split("_")[0]

    # --- YEARLY AND TOTAL DATA ---
    no_season_folder = os.path.join(wind_dir_folder, "no_season")
    if os.path.exists(no_season_folder):
        for file in glob.glob(os.path.join(no_season_folder, "weibull_results_*.csv")):
            df = pd.read_csv(file)
            df['wind_dir'] = pd.to_numeric(df['Wind direction'], errors='coerce')
            df['prob'] = pd.to_numeric(df['wind_dir_probability'], errors='coerce')
            df['rmse'] = pd.to_numeric(df['RMSE'], errors='coerce')
            df['ks'] = pd.to_numeric(df['K-S'], errors='coerce')

            label = os.path.splitext(os.path.basename(file))[0].replace("weibull_results_", "")
            df['year'] = label
            yearly_data.setdefault(n_dirs, []).append(df[['wind_dir', 'prob', 'rmse', 'ks', 'year']])

    # --- SEASONAL DATA ---
    season_folder = os.path.join(wind_dir_folder, "season")
    if os.path.exists(season_folder):
        for file in glob.glob(os.path.join(season_folder, "weibull_results_*.csv")):
            df = pd.read_csv(file)
            if 'season' in df.columns:
                df['wind_dir'] = pd.to_numeric(df['Wind direction'], errors='coerce')
                df['prob'] = pd.to_numeric(df['Probability'], errors='coerce')
                df['rmse'] = pd.to_numeric(df['RMSE'], errors='coerce')
                df['ks'] = pd.to_numeric(df['K-S'], errors='coerce')
                df['n_dirs'] = n_dirs
                seasonal_data.append(df[['wind_dir', 'prob', 'rmse', 'ks', 'season', 'n_dirs']])

# --- COMBINE YEARLY DATA ---
for n_dirs, dfs in yearly_data.items():
    combined = pd.concat(dfs, ignore_index=True)
    for metric, ylabel in [('rmse', 'RMSE'), ('ks', 'K-S Statistic')]:
        plt.figure(figsize=(10, 6))
        if (combined['year']=='tot').all():
            sns.scatterplot(data=df, x='prob', y=metric, color='tab:blue', legend=False)
        else:
            sns.scatterplot(data=combined, x='prob', y=metric, hue='year', palette='tab10')
        plt.xlabel('Wind Direction Probability (%)')
        plt.ylabel(ylabel)
        plt.title(f'{ylabel} vs Wind Direction Probability ({n_dirs} directions)')
        plt.grid(True)
        plt.xlim((0,max(combined['prob'])*1.2))
        if metric == 'rmse':
            plt.ylim((0,0.035))
        else:
            plt.ylim((0,0.2))
        plt.legend(loc='upper right')
        plt.tight_layout()
        save_path = os.path.join(output_folder, f"{metric}_vs_prob_yearly_{n_dirs}wd.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    
# --- COMBINE SEASONAL DATA ---
if seasonal_data:
    combined_seasonal = pd.concat(seasonal_data, ignore_index=True)
    for n_dirs in combined_seasonal['n_dirs'].unique():
        df_subset = combined_seasonal[combined_seasonal['n_dirs'] == n_dirs]
        for metric, ylabel in [('rmse', 'RMSE'), ('ks', 'K-S Statistic')]:
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df_subset, x='prob', y=metric, hue='season', palette='tab10')
            plt.xlabel('Wind Direction Probability (%)')
            plt.ylabel(ylabel)
            plt.title(f'{ylabel} vs Wind Direction Probability (Seasonal, {n_dirs} directions)')
            plt.grid(True)
            plt.legend(loc='upper right')
            plt.xlim((0,max(combined['prob'])*1.2))
            if metric == 'rmse':
                plt.ylim((0,0.035))
            else:
                plt.ylim((0,0.2))
            plt.tight_layout()
            save_path = os.path.join(output_folder, f"{metric}_vs_prob_seasonal_{n_dirs}wd.png")
            plt.savefig(save_path, dpi=300)
            plt.close()

