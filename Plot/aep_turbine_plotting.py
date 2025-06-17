# -*- coding: utf-8 -*-
"""
Created on Sun May 25 15:40:51 2025

@author: erica
"""

import os
from datetime import datetime
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.size': 15})

parent_folder = "D:/Thesis/Data/Penmanshiel/2025-06-01/"
output_folder = f"D:/Thesis/Figures/Penmanshiel/AEP/{datetime.now().strftime('%Y-%m-%d')}/Turbine"
os.makedirs(output_folder, exist_ok=True)


# Main processing
csv_files = glob.glob(os.path.join(parent_folder, "aep_results_*winddirections_no_season_frac.csv"))
aggregated_errors = []
turbine_results = {}

for file in csv_files:
    basename = os.path.basename(file)
    turbine_df = pd.read_csv(file)
    turbine_df = turbine_df[turbine_df['turbine_id'] != 'Total'].copy()
    n = int(basename.split("_")[2].replace("winddirections", ""))
    turbine_results[n] = turbine_df #.sort_values('turbine_id')  # Store sorted for consistency
    # Create a bar plot comparing real and adjusted AEP per wind turbine
    plt.figure(figsize=(12, 6))
    turbine_ids = turbine_df['turbine_id'].astype(str)
    
    plt.bar(turbine_ids, turbine_df['aep_real'], width=0.4, label='AEP Real', align='center')
    plt.bar(turbine_ids, turbine_df['aep_pywake_adj'], width=0.4, label='AEP PyWake Adj', align='edge')
    
    plt.xlabel('Wind Turbine ID')
    plt.ylabel('AEP (GWh)')
    #plt.title(f'Real vs Adjusted AEP per Wind Turbine ({n} wind directions)')
    plt.legend()
    plt.grid(axis='y')
    plt.tight_layout()
    plt.ylim((4,7))
    # Save plot
    plot_path = os.path.join(output_folder, 'aep_turbine_{n}winddirections.png')
    plt.savefig(plot_path)
    plt.show()


#%%

# Set up plot style

plt.rcParams.update({'font.size': 20})
colors = {
    '12': '#1f77b4',  # blue
    '36': '#ff7f0e',  # orange
    'real': '#7f7f7f'  # grey
}

bar_width = 0.2
offsets = {'12': -0.2, '36': 0, 'real': 0.2}

# Prepare figure
plt.figure(figsize=(14, 7))
turbine_ids = turbine_results[12]['turbine_id'].astype(str).values
x = np.arange(len(turbine_ids))

# Plot bars for each method
for key, df in turbine_results.items():
    label = f'{key} directions'
    plt.bar(x + offsets[str(key)], df['aep_pywake_adj'], width=bar_width, color=colors[str(key)], label=label)

# Plot real AEP
plt.bar(x + offsets['real'], turbine_results[12]['aep_real'], width=bar_width,
        color=colors['real'], label='AEP Real', hatch='//', alpha=0.7)

# Labels and formatting
plt.xticks(x, turbine_ids)
plt.xlabel('Wind Turbine ID')
plt.ylabel('AEP (MWh)')
#plt.title('Adjusted vs Real AEP per Wind Turbine\nComparison by Wind Direction Resolution', fontsize=15)
plt.ylim(0, 8)
#plt.legend(title='Resolution', fontsize=15)
plt.legend(loc='upper right')
plt.grid(axis='y')
plt.ylim((4,7))
plt.tight_layout()

# Add value labels
# for i, tid in enumerate(turbine_ids):
#     for key, df in turbine_results.items():
#         val = df['aep_pywake_adj'].iloc[i]
#         plt.text(x[i] + offsets[str(key)], val + 0.05, f'{val:.2f}', ha='center', fontsize=8)
#     real_val = turbine_results[12]['aep_real'].iloc[i]
#     plt.text(x[i] + offsets['real'], real_val + 0.05, f'{real_val:.2f}', ha='center', fontsize=8)

# Save plot
final_comparison_path = os.path.join(output_folder, "turbine_aep_comparison_final.png")
plt.savefig(final_comparison_path, dpi=300)
plt.show()

#%% with 48

# Set up plot style
#plt.style.use('seaborn-whitegrid')
# colors = {
#     '12': '#1f77b4',  # blue
#     '36': '#ff7f0e',  # orange
#     '48': '#2ca02c',  # green
#     'real': '#7f7f7f'  # grey
# }
# bar_width = 0.2
# offsets = {'12': -0.3, '36': -0.1, '48': 0.1, 'real': 0.3}

# # Prepare figure
# plt.figure(figsize=(14, 7))
# turbine_ids = turbine_results[12]['turbine_id'].astype(str).values
# x = np.arange(len(turbine_ids))

# # Plot bars for each method
# for key, df in turbine_results.items():
#     label = f'{key} directions'
#     plt.bar(x + offsets[str(key)], df['aep_pywake_adj'], width=bar_width, color=colors[str(key)], label=label)

# # Plot real AEP
# plt.bar(x + offsets['real'], turbine_results[12]['aep_real'], width=bar_width,
#         color=colors['real'], label='AEP Real', hatch='//', alpha=0.7)

# # Labels and formatting
# plt.xticks(x, turbine_ids)
# plt.xlabel('Wind Turbine ID', fontsize=13)
# plt.ylabel('AEP (GWh)', fontsize=13)
# #plt.title('Adjusted vs Real AEP per Wind Turbine\nComparison by Wind Direction Resolution', fontsize=15)
# plt.ylim(0, 8)
# #plt.legend(title='Resolution', fontsize=15)
# plt.legend(loc='upper right')
# plt.grid(axis='y')
# plt.ylim((4,7))
# plt.tight_layout()

# # Add value labels
# # for i, tid in enumerate(turbine_ids):
# #     for key, df in turbine_results.items():
# #         val = df['aep_pywake_adj'].iloc[i]
# #         plt.text(x[i] + offsets[str(key)], val + 0.05, f'{val:.2f}', ha='center', fontsize=8)
# #     real_val = turbine_results[12]['aep_real'].iloc[i]
# #     plt.text(x[i] + offsets['real'], real_val + 0.05, f'{real_val:.2f}', ha='center', fontsize=8)

# # Save plot
# final_comparison_path = os.path.join(output_folder, "turbine_aep_comparison_final.png")
# plt.savefig(final_comparison_path, dpi=300)
# plt.show()
