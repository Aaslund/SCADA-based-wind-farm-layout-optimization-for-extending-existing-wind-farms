# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 10:20:15 2025

@author: erica
"""

import pandas as pd

n=12
season_bool=0
consensus_bool = 1
# Load the Weibull results from CSV
load_folder = 'D:/Thesis/Data/Penmanshiel/2025-06-01/'
load_folder = load_folder+f'/Weibull/{n}_winddirections/'
if consensus_bool:
    load_folder = load_folder+'consensus/'
else:
    load_folder = load_folder+'consensus/'
if season_bool:
    load_folder = load_folder+'season/'
else:
    load_folder = load_folder+'no_season/'
save_folder = load_folder
df = pd.read_csv(load_folder+"weibull_results_tot.csv")

# Define thresholds
RMSE_THRESHOLD = 0.03
TAIL_ERROR_THRESHOLD = 0.04

# Apply criteria
df['RMSE_OK'] = df['rmse'] < RMSE_THRESHOLD

# Check for existence of tail error column
if 'tail error' in df.columns:
    df['TAIL_OK'] = df['tail error'] < TAIL_ERROR_THRESHOLD
else:
    df['TAIL_OK'] = True  # Mark as not evaluated

# Initialize summary collector
seasonal_summaries = []

if season_bool and 'Season' in df.columns:
    grouped = df.groupby('Season')
else:
    grouped = [('All Data', df)]  # Single group if no season

for season_label, group in grouped:
    rmse_pass_prob = group[group['RMSE_OK']]['wind_dir_probability'].sum()
    tail_pass_prob = group[group['TAIL_OK'] == True]['wind_dir_probability'].sum() if group['TAIL_OK'].notna().any() else None

    seasonal_summaries.append({
        'Season': season_label,
        'Criteria': f'RMSE < {RMSE_THRESHOLD:.3f}',
        'Bins Passing': group['RMSE_OK'].sum(),
        'Cumulative Wind Probability': rmse_pass_prob,
        'Satisfied': rmse_pass_prob >= 0.80
    })

    seasonal_summaries.append({
        'Season': season_label,
        'Criteria': f'Tail Error < {TAIL_ERROR_THRESHOLD:.3f}' if tail_pass_prob is not None else 'Tail Error (not available)',
        'Bins Passing': group['TAIL_OK'].sum() if tail_pass_prob is not None else 'N/A',
        'Cumulative Wind Probability': tail_pass_prob if tail_pass_prob is not None else 'N/A',
        'Satisfied': tail_pass_prob >= 0.80 if tail_pass_prob is not None else 'N/A'
    })

# Create DataFrame from results
summary_df = pd.DataFrame(seasonal_summaries)

# Save
summary_df.to_csv(save_folder + "weibull_fit_summary.csv", index=False)
df[['season', 'wind_dir_bin', 'wind_dir_probability', 'rmse', 'RMSE_OK', 'tail error', 'TAIL_OK']].to_csv(
    save_folder + "weibull_fit_detailed.csv", index=False
)
