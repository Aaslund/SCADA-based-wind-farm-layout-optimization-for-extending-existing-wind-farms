# Final unified script to process all AEP files, generate plots, and save error summaries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from datetime import datetime

# Setup
parent_folder = "D:/Thesis/Data/Penmanshiel/2025-06-01/"
output_folder = f"D:/Thesis/Figures/Penmanshiel/AEP/{datetime.now().strftime('%Y-%m-%d')}/"
os.makedirs(output_folder, exist_ok=True)
plt.rcParams.update({'font.size': 15})

# Grouping function to convert higher-resolution bins to 12-bin format
def map_to_12(wd, original_n):
    if original_n == 36:
        mapping = {350: 0, 0: 0, 10: 0, 20: 30, 30: 30, 40: 30, 50: 60, 60: 60, 70: 60,
                   80: 90, 90: 90, 100: 90, 110: 120, 120: 120, 130: 120, 140: 150,
                   150: 150, 160: 150, 170: 180, 180: 180, 190: 180, 200: 210, 210: 210,
                   220: 210, 230: 240, 240: 240, 250: 240, 260: 270, 270: 270, 280: 270,
                   290: 300, 300: 300, 310: 300, 320: 330, 330: 330, 340: 330}
        return wd.map(mapping)
    else:
        return (np.round(wd / 30) * 30) % 360

# Plot functions
def plot_aep(df, title, suffix, folder):
    plt.figure(figsize=(12, 5))
    plt.bar(df['wind_dir'] - 1.5, df['aep_real'], width=3, label='AEP Real')
    plt.bar(df['wind_dir'] + 1.5, df['aep_pywake_adj'], width=3, label='AEP PyWake Adj')
    plt.xlabel('Wind Direction (°)')
    plt.ylabel('AEP (MWh)')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"aep_comparison_{suffix}.png"))
    plt.close()

def plot_error(df, title, suffix, folder):
    plt.figure(figsize=(12, 4))
    plt.bar(df['wind_dir'], df['error'], width=3, color='coral')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Wind Direction (°)')
    plt.ylabel('Error (MWh)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"aep_error_bar_{suffix}.png"))
    plt.close()

def plot_polar_error(df, title, suffix, folder):
    angles = np.deg2rad(df['wind_dir'].values)
    errors = df['error'].values
    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.bar(angles, errors, width=np.deg2rad(360 / len(df)), color='steelblue', alpha=0.8)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(title, y=1.1)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"aep_error_polar_{suffix}.png"))
    plt.close()

def plot_relative_error(df, title, suffix, folder):
    plt.figure(figsize=(8, 5))
    plt.scatter(df['prob'], df['relative error'], c='purple')
    plt.xlabel('Wind Direction Probability (%)')
    plt.ylabel('Relative Error (%)')
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(folder, f"aep_relative_error_{suffix}.png"))
    plt.close()

# Main processing
csv_files = glob.glob(os.path.join(parent_folder, "wind_aep_results_*winddirections_*.csv"))
aggregated_errors = []

for file in csv_files:
    basename = os.path.basename(file)
    df = pd.read_csv(file)

    if 'wind_dir' not in df.columns or df.empty:
        continue

    is_seasonal = 'season' in df.columns
    is_yearly = 'year' in df.columns
    original_n = int(basename.split("_")[3].replace("winddirections", ""))

    # Sanitize
    df = df[df['wind_dir'] != 'Total'].copy()
    df['wind_dir'] = pd.to_numeric(df['wind_dir'], errors='coerce')
    df['aep_real'] = pd.to_numeric(df['aep_real'], errors='coerce')
    df['aep_pywake_adj'] = pd.to_numeric(df['aep_pywake_adj'], errors='coerce')

    group_col = 'season' if is_seasonal else 'year' if is_yearly else None

    if group_col:
        for g, group_df in df.groupby(group_col):
            suffix = f"{original_n}wd_{group_col}_{g}"
            folder = os.path.join(output_folder, suffix)
            os.makedirs(folder, exist_ok=True)

            df_g = group_df.groupby('wind_dir').agg({
                'aep_real': 'sum',
                'aep_pywake_adj': 'sum',
                'error': 'sum',
                'prob': 'sum'
            }).reset_index()
            df_g['relative error'] = np.abs(df_g['error']) / df_g['aep_real'] * 100

            plot_aep(df_g, f"AEP Real vs PyWake - {suffix}", suffix, folder)
            plot_error(df_g, f"AEP Error per Direction - {suffix}", suffix, folder)
            plot_polar_error(df_g, f"AEP Error Polar - {suffix}", suffix, folder)
            plot_relative_error(df_g, f"Relative Error vs Frequency - {suffix}", suffix, folder)

            aggregated_errors.append({
                'source': suffix,
                'mean_relative_error': df_g['relative error'].mean(),
                'total_error': df_g['error'].sum()
            })

            # Grouped to 12
            if original_n > 12:
                group_df['wind_dir_12'] = map_to_12(group_df['wind_dir'], original_n)
                df_12 = group_df.groupby('wind_dir_12').agg({
                    'aep_real': 'sum',
                    'aep_pywake_adj': 'sum',
                    'error': 'sum'
                }).reset_index().rename(columns={'wind_dir_12': 'wind_dir'})
                df_12['relative error'] = np.abs(df_12['error']) / df_12['aep_real'] * 100
                df_12['prob'] = df_12['aep_real'] / df_12['aep_real'].sum() * 100
                suffix_12 = f"{original_n}wd_grouped12_{group_col}_{g}"
                folder_12 = os.path.join(output_folder, suffix_12)
                os.makedirs(folder_12, exist_ok=True)

                plot_aep(df_12, f"AEP Real vs PyWake - {suffix_12}", suffix_12, folder_12)
                plot_error(df_12, f"AEP Error per Direction - {suffix_12}", suffix_12, folder_12)
                plot_polar_error(df_12, f"AEP Error Polar - {suffix_12}", suffix_12, folder_12)
                plot_relative_error(df_12, f"Relative Error vs Frequency - {suffix_12}", suffix_12, folder_12)

                aggregated_errors.append({
                    'source': suffix_12,
                    'mean_relative_error': df_12['relative error'].mean(),
                    'total_error': df_12['error'].sum()
                })

    else:
        suffix = f"{original_n}wd_total"
        folder = os.path.join(output_folder, suffix)
        os.makedirs(folder, exist_ok=True)

        df_g = df.groupby('wind_dir').agg({
            'aep_real': 'sum',
            'aep_pywake_adj': 'sum',
            'error': 'sum'
        }).reset_index()
        df_g['relative error'] = np.abs(df_g['error']) / df_g['aep_real'] * 100
        df_g['prob'] = df_g['aep_real'] / df_g['aep_real'].sum() * 100

        plot_aep(df_g, f"AEP Real vs PyWake - {suffix}", suffix, folder)
        plot_error(df_g, f"AEP Error per Direction - {suffix}", suffix, folder)
        plot_polar_error(df_g, f"AEP Error Polar - {suffix}", suffix, folder)
        plot_relative_error(df_g, f"Relative Error vs Frequency - {suffix}", suffix, folder)

        aggregated_errors.append({
            'source': suffix,
            'mean_relative_error': df_g['relative error'].mean(),
            'total_error': df_g['error'].sum()
        })

        if original_n > 12:
            df['wind_dir_12'] = map_to_12(df['wind_dir'], original_n)
            df_12 = df.groupby('wind_dir_12').agg({
                'aep_real': 'sum',
                'aep_pywake_adj': 'sum',
                'error': 'sum'
            }).reset_index().rename(columns={'wind_dir_12': 'wind_dir'})
            df_12['relative error'] = np.abs(df_12['error']) / df_12['aep_real'] * 100
            df_12['prob'] = df_12['aep_real'] / df_12['aep_real'].sum() * 100
            suffix_12 = f"{original_n}wd_grouped12_total"
            folder_12 = os.path.join(output_folder, suffix_12)
            os.makedirs(folder_12, exist_ok=True)

            plot_aep(df_12, f"AEP Real vs PyWake - {suffix_12}", suffix_12, folder_12)
            plot_error(df_12, f"AEP Error per Direction - {suffix_12}", suffix_12, folder_12)
            plot_polar_error(df_12, f"AEP Error Polar - {suffix_12}", suffix_12, folder_12)
            plot_relative_error(df_12, f"Relative Error vs Frequency - {suffix_12}", suffix_12, folder_12)

            aggregated_errors.append({
                'source': suffix_12,
                'mean_relative_error': df_12['relative error'].mean(),
                'total_error': df_12['error'].sum()
            })
#%%
comparison_files = {
    '12': 'wind_aep_results_12winddirections_no_season.csv',
    '12 season': 'wind_aep_results_12winddirections_season.csv',
    '36': 'wind_aep_results_36winddirections_no_season.csv',
    #'48': 'wind_aep_results_48winddirections_no_season.csv',
    '36 (12)': 'wind_aep_results_36winddirections_no_season.csv',
    #'12_year': 'wind_aep_results_12winddirections_year.csv'
    #'48_grouped12': 'wind_aep_results_48winddirections_no_season.csv',
}


dfs = {}
for key, filename in comparison_files.items():
    path = os.path.join(parent_folder, filename)
    if not os.path.exists(path):
        continue
    df = pd.read_csv(path)
    df = df[df['wind_dir'] != 'Total'].copy()
    df['wind_dir'] = pd.to_numeric(df['wind_dir'], errors='coerce')
    df['aep_real'] = pd.to_numeric(df['aep_real'], errors='coerce')
    df['aep_pywake_adj'] = pd.to_numeric(df['aep_pywake_adj'], errors='coerce')
    if key == '36 (12)':
        df['wind_dir'] = map_to_12(df['wind_dir'], 36)
        df = df.groupby('wind_dir')[['aep_real', 'aep_pywake_adj']].sum().reset_index()
    if key == '48 (12)':
        df['wind_dir'] = map_to_12(df['wind_dir'], 48)
        df = df.groupby('wind_dir')[['aep_real', 'aep_pywake_adj']].sum().reset_index()
    if key == '12 season':
        df = df.groupby('wind_dir')[['aep_real', 'aep_pywake_adj']].sum().reset_index()
    if key == '12 year':
        df = df.groupby('wind_dir')[['aep_real', 'aep_pywake_adj']].sum().reset_index()
        factor = 365 / (3*365 + 366)
        df[['aep_real', 'aep_pywake_adj']] = df[['aep_real', 'aep_pywake_adj']] * factor
     
    dfs[key] = df.sort_values('wind_dir')

# Create comparison plot
plt.figure(figsize=(12, 6))
for label, df in dfs.items():
    if label in ['12', '36','48']:
        plt.plot(df['wind_dir'], df['aep_real'], linestyle='--', label=f"Real - {label}", alpha=0.6)
    plt.plot(df['wind_dir'], df['aep_pywake_adj'], marker='o', linestyle='-', label=f"PyWake - {label}")
plt.xlabel('Wind Direction (°)')
plt.ylabel('AEP (MWh)')
#plt.title('Comparison of PyWake AEP Predictions\n12 vs 36 vs 36 grouped to 12 (no season)')
plt.legend()
plt.grid(True)
plt.tight_layout()

comparison_plot_path = os.path.join(output_folder, "comparison_aep_prediction_profiles_48.png")
plt.savefig(comparison_plot_path)
plt.close()
