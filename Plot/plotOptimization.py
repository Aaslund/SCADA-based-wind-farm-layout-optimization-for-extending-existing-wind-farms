# -*- coding: utf-8 -*-
"""
Created on Sun Jun  1 19:15:05 2025

@author: erica
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import numpy as np

load_folder = 'D:/Thesis/Data/Penmanshiel/2025-06-01/Optimization/Whole_farm/'
season_txt = 'no_season'
plt.rcParams.update({'font.size': 12})


n=12
iterations=501
# Load uploaded CSV files

figure_folder = f"D:/Thesis/Figures/Penmanshiel/Optimization/Whole_farm/{datetime.now().strftime('%Y-%m-%d')}/{n}winddirections/iter{iterations}"
os.makedirs(figure_folder, exist_ok=True)

opt_results_path = load_folder + f"/optimization_{n}winddirections_no_season_iter{iterations}.csv"
turbine_positions_path = load_folder + f"/turbine_positions_{n}winddirections_no_season_iter{iterations}.csv"

turbine_positions_path12 = load_folder + f"/turbine_positions_12winddirections_no_season_iter{iterations}.csv"
turbine_positions_path36 = load_folder + f"/turbine_positions_36winddirections_no_season_iter{iterations}.csv"

opt_df = pd.read_csv(opt_results_path)
turbine_df = pd.read_csv(turbine_positions_path)
turbine_df_unique = turbine_df.drop_duplicates(subset='Iteration').reset_index(drop=True)

# Keep only the relevant columns (optional)
turbine_df_unique = turbine_df_unique[['Iteration', 'AEP']]
turbine_df12 = pd.read_csv(turbine_positions_path12)
turbine_df_unique12 = turbine_df12.drop_duplicates(subset='Iteration').reset_index(drop=True)
turbine_df_unique12 = turbine_df_unique[['Iteration', 'AEP']]

turbine_df36 = pd.read_csv(turbine_positions_path36)
turbine_df_unique36 = turbine_df36.drop_duplicates(subset='Iteration').reset_index(drop=True)
turbine_df_unique36 = turbine_df_unique36[['Iteration', 'AEP']]

# Plot 1: AEP Convergence Plot
fig1, ax1 = plt.subplots(constrained_layout=True)
ax1.plot(turbine_df_unique12["Iteration"], turbine_df_unique12["AEP"]*1e3,label='12 directions')
ax1.plot(turbine_df_unique36["Iteration"], turbine_df_unique36["AEP"]*1e3,label='36 directions')
#ax1.set_title("AEP Convergence")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("AEP (MWh)")
ax1.legend(loc='lower right')
#ax1.grid(True)
aep_conv_path = os.path.join(figure_folder, f"aep_convergence_{n}winddirections_iter{iterations}.png")
fig1.savefig(aep_conv_path)

# Plot 2: Relative Increase vs Iteration
fig2, ax2 = plt.subplots(constrained_layout=True)
ax2.plot(turbine_df_unique["Iteration"][1:-1], turbine_df_unique["AEP"].values[1:-1]/turbine_df_unique["AEP"].values[0:-2]*100-100,  color='orange')
ax2.set_title("Relative AEP Increase")
ax2.set_xlabel("Iteration")
ax2.set_ylabel("Relative Increase [%]")
#ax2.grid(True)
rel_inc_path = os.path.join(figure_folder, f"relative_increase_{n}winddirections_iter{iterations}.png")
fig2.savefig(rel_inc_path)


#%%
# Load turbine positions and extract latest iteration (optimized positions)
latest_iter = turbine_df["Iteration"].max()
latest_positions = turbine_df[turbine_df["Iteration"] == latest_iter]

# If available, extract initial positions from iteration 0
initial_positions = turbine_df[turbine_df["Iteration"] == 0]

# Define and shift boundary
boundary = np.array([
    (0.0, 0.0),
    (759.3000118067339, -838.5008776575139),
    (1404.4887297180828, -369.7348857613972),
    (2015.077636388391, -32.399901569969394),
    (2152.4804284290312, 105.32498167009027),
    (1030.2364721134584, 1024.6572692781397),
    (388.8327571837189, 960.461527003533),
    (112.57349730534965, 830.3425848119384)
]) - np.array([164.31855339, 32.0470227])
closed_boundary = np.vstack([boundary, boundary[0]])

# Plot 1: Only optimized turbines with boundary
fig1, ax1 = plt.subplots(constrained_layout=True)
ax1.fill(closed_boundary[:, 0], closed_boundary[:, 1], color='green', alpha=0.2, label='Inclusion')
ax1.plot(closed_boundary[:, 0], closed_boundary[:, 1], color='green')
ax1.scatter(latest_positions["x"], latest_positions["y"], facecolors='grey', edgecolors='black',
            marker='o', label='Optimized position')
ax1.set_title("Optimized Turbine Positions")
ax1.set_xlabel("x [m]")
ax1.set_ylabel("y [m]")
ax1.set_xlim(-100, 2500)
ax1.set_ylim(-1100, 1100)
ax1.set_xticks([0, 500, 1000, 1500, 2000, 2500])
ax1.set_yticks([-1000, -500, 0, 500, 1000])
ax1.axis('equal')
ax1.legend(loc='lower right', fontsize=12)
plot1_path = os.path.join(figure_folder, f"optimized_positions_{n}winddirections_iter{iterations}.png")
fig1.savefig(plot1_path)

#%%
# Plot 2: Optimized + Initial turbines
fig2, ax2 = plt.subplots(constrained_layout=True)
iteration = latest_iter
# Get the increase/expansion factor if it exists, otherwise default to 1
# Compute percentage delta safely
aep=turbine_df_unique["AEP"]
cost = aep[len(aep)-1]
cost0 = aep[0]
delta = ((cost - cost0) / cost0 * 100) if cost0 else 0
if np.sign(delta) == -1:
    sign=''
else:
    sign='+'
ax2.fill(closed_boundary[:, 0], closed_boundary[:, 1], color='green', alpha=0.2, label='Inclusion zone')
ax2.plot(closed_boundary[:, 0], closed_boundary[:, 1], color='green')
ax2.scatter(initial_positions["x"], initial_positions["y"], color='grey', alpha=0.4,
            marker='^', label='Initial position')
ax2.scatter(latest_positions["x"], latest_positions["y"], facecolors='grey', edgecolors='black',
            marker='o', label='Optimized position')
ax2.set_title(f"Iterations: {iteration} | AEP: {cost*1000:.2f} MWh  |  Δ: {sign}{delta:.1f}%", fontsize=12)
ax2.set_xlabel("x [m]")
ax2.set_ylabel("y [m]")
ax2.set_xlim(-100, 2500)
ax2.set_ylim(-1100, 1500)
ax2.set_xticks([0, 500, 1000, 1500, 2000, 2500])
ax2.set_yticks([-1000, -500, 0, 500, 1000, 1500])
ax2.axis('equal')
ax2.legend(loc='lower right', fontsize=11)
#fig2.tight_layout()
#fig2.subplots_adjust(bottom=0.15)
plot2_path = os.path.join(figure_folder, f"initial_optimized_positions_{n}winddirections_iter{iterations}.png")
fig2.savefig(plot2_path, pad_inches=0.3)