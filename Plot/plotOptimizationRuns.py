# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 15:36:51 2025

@author: erica
"""

# Re-attempt plotting without displaying the dataframe through ace_tools
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
import numpy as np
n=12
#load_folder = f'D:/Thesis/Data/Penmanshiel/2025-06-01/Optimization/Whole_farm/{n}winddirections/MultiStart_2025-06-03/'
#season_txt = 'no_season'
plt.rcParams.update({'font.size': 20})

# Set the base folder where the results are stored
base_folder = f'D:/Thesis/Data/Penmanshiel/2025-06-01/Optimization/Whole_farm/{n}winddirectionsMultiStart_2025-06-03/'  # Adjust to your actual folder prefix
#base_folder=r'D:\Thesis\Data\Penmanshiel\2025-06-01\Optimization\Whole_farm\12winddirectionsMultiStart_2025-06-03'
matching_dirs = glob(base_folder + '/run_*')
matching_dirs.sort()  # Ensure consistent ordering
x0 = np.array([164.31855339, 483.54455926, 394.90183711, 664.38319266, 938.66446991, 1194.93045332, 730.02368931, 1073.3739787, 1336.49699347, 1661.5203035, 1124.49003472, 1442.34463091, 1706.46503031, 1975.82170693]) - 164.31855339
y0 = np.array([32.04702276, -245.27312429, 414.66876535, 120.11340467, -140.41630846, -386.15709635, 633.94516069, 308.70000026, 90.98033389, -151.4246062, 721.56676289, 535.09287091, 315.37169586, 27.71042063]) - 32.0470227

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
# Plot AEP evolution
plt.figure(figsize=(9, 6))
for run_path in matching_dirs:
    run_id = os.path.basename(run_path)
    layout_path = os.path.join(run_path, 'layout_history.csv')
    if not os.path.exists(layout_path):
        continue
    df = pd.read_csv(layout_path)
    grouped = df.groupby('Iteration').first()  # Get AEP per iteration
    run_id=run_id.replace('_',' ')
    plt.plot(grouped.index, grouped['AEP'], label=run_id)

plt.xlabel('Iterations',fontsize=20)
plt.ylabel('AEP [MWh]',fontsize=20)
#plt.title('AEP Evolution Across Runs')
plt.legend(loc='lower right',fontsize=18)
#plt.grid(True)
plt.tight_layout()
plt.show()

plt.rcParams.update({'font.size': 15})
# Plot final turbine positions for each run
fig1, ax1 = plt.subplots(constrained_layout=True)
ax1.fill(closed_boundary[:, 0], closed_boundary[:, 1], color='green', alpha=0.2, label='Inclusion')
ax1.plot(closed_boundary[:, 0], closed_boundary[:, 1], color='green')
ax1.scatter(x0,y0,s=30,label='Initial position', color='grey', marker='^')
for run_path in matching_dirs:
    run_id = os.path.basename(run_path)
    layout_path = os.path.join(run_path, 'layout_history.csv')
    if not os.path.exists(layout_path):
        continue
    df = pd.read_csv(layout_path)
    final_positions = df[df['Iteration'] == df['Iteration'].max()]

    # Plot 1: Only optimized turbines with boundary
    run_id=run_id.replace('_',' ')
    ax1.scatter(final_positions['x'], final_positions['y'], s=20,label=run_id)

plt.xlabel('x [m]')
plt.ylabel('y [m]')
#plt.title('Final Turbine Positions for Each Run')
plt.legend(loc='lower right', fontsize=10)
#plt.grid(True)
plt.axis('equal')
plt.tight_layout()
plt.show()
