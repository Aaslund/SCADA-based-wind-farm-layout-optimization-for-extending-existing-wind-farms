# Full integrated version of the AEP comparison and plotting script

import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from datetime import datetime
import numpy as np
plt.rcParams.update({'font.size': 15})

def extract_n_season_year(filename):
    basename = os.path.basename(filename).split('.')[0]  # Remove extension first
    parts = basename.split("_")
    n = int(parts[3][:2])
    if parts[4] == 'no':
        season_txt = 'no_season'
    elif parts[4] == 'year':
        season_txt = 'year'
    else:
        season_txt = 'season'
    return n, season_txt

def map_36_to_12(wind_dir_36):
    mapping = {
        350: 0, 0: 0, 10: 0,
        20: 30, 30: 30, 40: 30,
        50: 60, 60: 60, 70: 60,
        80: 90, 90: 90, 100: 90,
        110: 120, 120: 120, 130: 120,
        140: 150, 150: 150, 160: 150,
        170: 180, 180: 180, 190: 180,
        200: 210, 210: 210, 220: 210,
        230: 240, 240: 240, 250: 240,
        260: 270, 270: 270, 280: 270,
        290: 300, 300: 300, 310: 300,
        320: 330, 330: 330, 340: 330
    }
    return wind_dir_36.map(mapping)

# Paths
parent_folder = "D:/Thesis/Data/Penmanshiel/2025-05-24/"
all_files = glob.glob(os.path.join(parent_folder, "wind_aep_results_*.csv"))
now = datetime.now()
date_folder = now.strftime("%Y-%m-%d")
figure_folder = f'D:/Thesis/Figures/Penmanshiel/AEP/{date_folder}/'
os.makedirs(figure_folder, exist_ok=True)

# Process files
for file in all_files:
    n, season_txt = extract_n_season_year(file)
    aep_df = pd.read_csv(file)

    df = aep_df[aep_df['wind_dir'] != 'Total'].copy()
    df['wind_dir'] = pd.to_numeric(df['wind_dir'], errors='coerce')

    if n == 36:
        df['wind_dir'] = map_36_to_12(df['wind_dir'])
        df = df.groupby('wind_dir')[['aep_real', 'aep_pywake_adj','prob']].sum().reset_index()
        df['error'] = df['aep_pywake_adj'] - df['aep_real']
        df['relative error'] = np.abs(df['error']) / df['aep_real'] * 100

    if season_txt == 'year':
        df = df.groupby('wind_dir')[['aep_real', 'aep_pywake_adj','prob']].sum().reset_index()
        df['error'] = df['aep_pywake_adj'] - df['aep_real']
        df['relative error'] = np.abs(df['error']) / df['aep_real'] * 100
    # Bar Plot of Real vs. Adjusted AEP per Direction
    plt.figure(figsize=(12, 5))
    plt.bar(df['wind_dir'] - 1.5, df['aep_real'], width=3, label='AEP Real')
    plt.bar(df['wind_dir'] + 1.5, df['aep_pywake_adj'], width=3, label='AEP PyWake Adj')
    plt.xlabel('Wind Direction (°)')
    plt.ylabel('AEP (GWh)')
    plt.title(f'Real vs Adjusted PyWake AEP per Direction ({n} bins grouped to 12) {season_txt}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    filename = f"AEP_comp_{n}_winddirections_{season_txt}.png"
    plt.savefig(os.path.join(figure_folder, filename))
    plt.show()

    # Error per Direction (Bar Plot)
    plt.figure(figsize=(12, 4))
    plt.bar(df['wind_dir'], df['error'], width=3, color='coral')
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Wind Direction (°)')
    plt.ylabel('Error (GWh)')
    plt.title(f'AEP Error per Wind Direction (Adjusted - Real) ({n} bins grouped to 12) {season_txt}')
    plt.grid(True)
    plt.tight_layout()
    filename = f"error_bar_{n}_winddirections_{season_txt}.png"
    plt.savefig(os.path.join(figure_folder, filename))
    plt.show()

    # Wind Rose Style Error Plot
    angles = np.deg2rad(df['wind_dir'].values)
    errors = df['error'].values

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    bars = ax.bar(angles, errors, width=np.deg2rad(360 / len(df)), color='steelblue', alpha=0.8)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_title(f'Directional AEP Error (Polar View) ({n} bins grouped to 12) {season_txt}', y=1.1)
    plt.tight_layout()
    filename = f"error_polar_{n}_winddirections_{season_txt}.png"
    plt.savefig(os.path.join(figure_folder, filename))
    plt.show()

    # Relative Error vs. Wind Frequency
    plt.figure(figsize=(8, 5))
    plt.scatter(df['prob'], df['relative error'], c='purple')
    plt.xlabel('Wind Direction Probability (%)')
    plt.ylabel('Relative Error (%)')
    plt.title(f'Relative Error vs. Wind Direction Frequency ({n} bins grouped to 12) {season_txt}')
    plt.grid(True)
    plt.tight_layout()
    filename = f"error_freq_{n}_winddirections_{season_txt}.png"
    plt.savefig(os.path.join(figure_folder, filename))
    plt.show()
