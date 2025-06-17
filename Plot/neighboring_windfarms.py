# -*- coding: utf-8 -*-
"""
Created on Mon May 12 14:03:11 2025

@author: erica
"""

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 15})

# Load the data
df = pd.read_csv('D:/Thesis/Data/repd-q4-jan-2025.csv',usecols=["Site Name", "Technology Type", "Installed Capacity (MWelec)", "Development Status", "X-coordinate", "Y-coordinate", "Operational"],encoding='windows-1252')

# Penmanshiel coordinates
penmanshiel_x = 380006/1e4
penmanshiel_y = 667129/1e4
search_radius = 10  # Half of 10,000 to get +/- 5000 in each direction

wind_farms = df[(df['Technology Type'] =='Wind Onshore') & ((df['Development Status'] =='Operational')|(df['Development Status'] =='Under Construction'))]
#%%
# Calculate distance from Penmanshiel
wind_farms['X-coordinate'] = pd.to_numeric(wind_farms['X-coordinate'], errors='coerce')/1e4
wind_farms['Y-coordinate'] = pd.to_numeric(wind_farms['Y-coordinate'], errors='coerce')/1e4
wind_farms['X_diff'] =  wind_farms['X-coordinate'] - penmanshiel_x
wind_farms['Y_diff'] =  wind_farms['Y-coordinate'] - penmanshiel_y
wind_farms.loc[:, 'X_diff'] = (wind_farms['X_diff'] / 0.3713) * 3.693
wind_farms.loc[:, 'Y_diff'] = (wind_farms['Y_diff'] / 0.3713) * 3.693
# Filter for wind farms within 10,000m square
nearby_wind_farms = wind_farms[
    (wind_farms['X_diff'].abs() <= search_radius) & 
    (wind_farms['Y_diff'].abs() <= search_radius)
]

# Select only the columns we want
result_df = nearby_wind_farms
mask = result_df['Site Name'] != 'Penmanshiel Wind Farm'
result_df.loc[mask, 'X_diff'] = result_df.loc[mask, 'X_diff'] - 2
result_df.loc[mask, 'Y_diff'] = result_df.loc[mask, 'Y_diff'] - 0.7
#%% Save to new CSV
result_df.to_csv('D:/Thesis/Data/penmanshiel_nearby_wind_farms.csv', index=False)
#%%
# Create the plot
plt.figure(figsize=(10, 10))
plt.rcParams.update({'font.size': 18})

result_df['color'] = 'blue'  # Default color
if (result_df['Development Status'] == 'Under Construction').any():
    result_df.loc[df['Development Status'] == 'Under Construction', 'color'] = 'magenta'


active = result_df[result_df['Development Status'] == 'Operational']
# Plot all wind farms
plt.scatter(
    active['X_diff'],
    active['Y_diff'],
    #s=pd.to_numeric(result_df['Installed Capacity (MWelec)']*2, errors='coerce'),  # Size by capacity
    s=active['Installed Capacity (MWelec)'].astype(float)*10,
    c=active['color'],
    alpha=0.6,
    label='Operational'
)
uc = result_df[result_df['Development Status'] == 'Under Construction']
plt.scatter(
    uc['X_diff'],
    uc['Y_diff'],
    #s=pd.to_numeric(result_df['Installed Capacity (MWelec)']*2, errors='coerce'),  # Size by capacity
    s=uc['Installed Capacity (MWelec)'].astype(float)*10,
    c=uc['color'],
    alpha=0.6,
    label='Under construction'
)

# Highlight Penmanshiel
plt.scatter(
    0,0,
    s=200,
    c='red',
    marker='x',
    label='Penmanshiel'
)

for _, row in active.iterrows():
    plt.annotate(
        f"{row['Site Name']}\n{row['Installed Capacity (MWelec)']}MW",
        xy=(row['X_diff'], row['Y_diff']),
        xytext=(-50, 15),  # Offset to avoid overlap
        textcoords='offset points',
        fontsize=15
    )
for _, row in uc.iterrows():
    plt.annotate(
        f"{row['Site Name']}\n{row['Installed Capacity (MWelec)']}MW",
        xy=(row['X_diff'], row['Y_diff']),
        xytext=(-50, -50),  # Offset to avoid overlap
        textcoords='offset points',
        fontsize=15
    )
# Add a square to show the search area
# plt.gca().add_patch(plt.Rectangle(
#     (- search_radius, - search_radius),
#     2*search_radius,
#     2*search_radius,
#     fill=False,
#     color='red',
#     linestyle='--',
#     label='Search Area'
# ))

#plt.title('Wind Farms Near Penmanshiel')
plt.xlabel('X Coordinate (km)')
plt.ylabel('Y Coordinate (km)')
plt.legend(loc='upper right')
plt.grid(True)
plt.tight_layout()

# Save and show the plot
plt.savefig('penmanshiel_wind_farms_map.png', dpi=300)
plt.show()
