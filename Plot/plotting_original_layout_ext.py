# -*- coding: utf-8 -*-
"""
Created on Mon May 26 13:48:01 2025

@author: erica
"""
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 12})

boundary = np.array([
    (0.0, 0.0),
    (759.3000118067339, -838.5008776575139),
    (1404.4887297180828, -369.7348857613972),
    (2015.077636388391, -32.399901569969394),
    (2152.4804284290312, 105.32498167009027),
    (1030.2364721134584, 1024.6572692781397),
    (388.8327571837189, 960.461527003533),
    (112.57349730534965, 830.3425848119384)
])

x=np.array([ 164.31855339,  483.54455926,  394.90183711,  664.38319266,
        938.66446991, 1194.93045332,  730.02368931, 1073.3739787 ,
       1336.49699347, 1661.5203035 , 1124.49003472, 1442.34463091,
       1706.46503031, 1975.82170693])
y=np.array([  32.04702276, -245.27312429,  414.66876535,  120.11340467,
       -140.41630846, -386.15709635,  633.94516069,  308.70000026,
         90.98033389, -151.4246062 ,  721.56676289,  535.09287091,
        315.37169586,   27.71042063])

boundary_closed = np.vstack([boundary, boundary[0]])
plt.plot(boundary_closed[:, 0], boundary_closed[:, 1], 'k-', label='Boundary')

# Plot wind turbines
plt.plot(x, y, 'bo', markersize=5, label='Wind Turbines',color='darkblue')

for i, (xi, yi) in enumerate(zip(x, y)):
    plt.text(xi-40, yi, f' {i+1}', color='darkblue', va='top', ha='right')
    
turbine_groups = [
    [0, 1],             # Turbines 1 and 2
    [2, 3, 4, 5],       # Turbines 3 (skipped), 4, 5, 6
    [6, 7, 8, 9],       # Turbines 7 to 10
    [10, 11, 12, 13]    # Turbines 11 to 14
]

# Draw lines between selected turbine groups
for group in turbine_groups:
    plt.plot(x[group], y[group], 'b--')

# Labels and legend
plt.xlabel("x [m]")
plt.ylabel("y [m]")
#plt.title("Wind Turbines and Site Boundary")
plt.legend(loc='lower right')
plt.axis('equal')
#plt.grid(True)
plt.tight_layout()
plt.show()
#%%

def calculate_bearing(x_start, y_start, x_end, y_end):
    dx = x_end - x_start
    dy = y_end - y_start
    angle_rad = np.arctan2(dx, dy)  # Angle from North (dy is Y-axis)
    angle_deg = np.degrees(angle_rad)
    bearing = (angle_deg + 360) % 360  # Ensure 0°-360°
    return bearing

# --- Line 1: Turbines [2, 3, 4, 5] ---
x_start1, y_start1 = x[2], y[2]  # Turbine 3 (index 2)
x_end1, y_end1 = x[5], y[5]      # Turbine 6 (index 5)
bearing1 = calculate_bearing(x_start1, y_start1, x_end1, y_end1)

# --- Line 2: Turbines [6, 7, 8, 9] ---
x_start2, y_start2 = x[6], y[6]  # Turbine 7 (index 6)
x_end2, y_end2 = x[9], y[9]      # Turbine 10 (index 9)
bearing2 = calculate_bearing(x_start2, y_start2, x_end2, y_end2)

print(f"Bearing of Line [2,3,4,5] (Turbines 3-6): {bearing1:.1f}° from North")
print(f"Bearing of Line [6,7,8,9] (Turbines 7-10): {bearing2:.1f}° from North")