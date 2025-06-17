# -*- coding: utf-8 -*-
"""
Created on Tue May 27 13:50:00 2025

@author: erica
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
plt.rcParams.update({'font.size': 20})

import sys
import os
from pathlib import Path

# Get the parent directory (A) and add it to sys.path
parent_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(parent_dir)

from support_functions.weight_function import weight_function

# def weight_function(x, y, wind_direction, turbine_diameter, wake_angle=20):
#     """
#     Calculate weights for turbines based on wake effects, considering the entire rotor width.

#     Parameters:
#         x (np.array): X positions of turbines
#         y (np.array): Y positions of turbines
#         wind_direction (float): Wind direction in degrees (meteorological convention)
#         turbine_diameter (float): Diameter of the wind turbine
#         wake_angle (float): Wake expansion angle in degrees (default: 5°)

#     Returns:
#         np.array: Weights for each turbine (1 = unaffected, 0 = in wake)
#     """
#     num_turbines = len(x)
#     weights = np.ones(num_turbines)  # All turbines start unaffected

#     # Convert wind direction to radians (FIX: Rotate by -90° to match Cartesian system)
#     wind_dir_rad = np.deg2rad(270-wind_direction)
    
#     # Wind vector (normalized)
#     wind_vector = np.array([np.cos(wind_dir_rad), np.sin(wind_dir_rad)])

#     for i in range(num_turbines):
#         for j in range(num_turbines):
#             if i == j:
#                 continue  # Skip self-check

#             # Compute displacement from turbine j to turbine i
#             dx, dy = x[i] - x[j], y[i] - y[j]
#             distance = np.sqrt(dx**2 + dy**2)
#             if distance == 0:
#                 continue

#             # Compute perpendicular distance from wake centerline
#             cross_product = dx * wind_vector[1] - dy * wind_vector[0]
#             # **Wake Region Definition**
#             projection = np.dot([dx, dy], wind_vector)  # Projection of turbine i onto wind vector (to check if it's downwind)
#             if projection > 0:
#                 lateral_distance = abs(cross_product)  # Distance from wake axis
#                 if lateral_distance < turbine_diameter / 2 + projection * np.tan(np.deg2rad(wake_angle)):  # Turbine j is upwind of turbine i
#                     weights[i] = 0  # Mark as affected by wake
#                     if i==0:
#                         print(j)
#                         print(f"lateral distance: {lateral_distance}")
#                         print(f"critertia: {turbine_diameter / 2 + projection * np.tan(np.deg2rad(wake_angle))}")
#     return weights


def plot_wake_region(ax, x, y, wind_direction, turbine_diameter, wake_angle=20):
    """
    Plot the wake region as a light blue polygon behind the turbines for a single wind direction.
    
    Parameters:
    - ax: matplotlib axis object
    - x, y: arrays of turbine coordinates
    - wind_direction: wind direction in degrees (meteorological convention)
    - turbine_diameter: diameter of turbines in meters
    - wake_angle: wake expansion angle in degrees (default=5)
    """
    # Convert wind direction to radians (adjust for Cartesian coordinates)
    wind_dir_rad = np.deg2rad(270 - wind_direction)  # 270-wind_direction converts to Cartesian
    wind_vector = np.array([np.cos(wind_dir_rad), np.sin(wind_dir_rad)])

    # Distance to extend the wake region
    wake_length = 1000  # Extend wake to plot limits

    # Calculate the wake region for each turbine
    for turbine_x, turbine_y in zip(x, y):
        # Calculate the wake edges at turbine position
        left_edge = [
            turbine_x - (turbine_diameter / 2) * wind_vector[1],
            turbine_y + (turbine_diameter / 2) * wind_vector[0]
        ]
        right_edge = [
            turbine_x + (turbine_diameter / 2) * wind_vector[1],
            turbine_y - (turbine_diameter / 2) * wind_vector[0]
        ]

        # Calculate wake edges at end of wake region
        left_edge_end = [
            left_edge[0] + wake_length * wind_vector[0] - wake_length * np.tan(np.deg2rad(wake_angle)) * wind_vector[1],
            left_edge[1] + wake_length * wind_vector[1] + wake_length * np.tan(np.deg2rad(wake_angle)) * wind_vector[0]
        ]
        right_edge_end = [
            right_edge[0] + wake_length * wind_vector[0] + wake_length * np.tan(np.deg2rad(wake_angle)) * wind_vector[1],
            right_edge[1] + wake_length * wind_vector[1] - wake_length * np.tan(np.deg2rad(wake_angle)) * wind_vector[0]
        ]

        # Define the polygon vertices
        wake_polygon = np.array([left_edge, right_edge, right_edge_end, left_edge_end])

        # Create and add polygon patch
        wake_patch = Polygon(wake_polygon, closed=True, color='lightblue', alpha=0.3)
        ax.add_patch(wake_patch)
    return ax
    
# Turbine coordinates
x = np.array([0., 319.22363416, 230.58157057, 500.06092411,
              774.34016385, 1030.60424373, 565.70093313, 909.048672,
              1172.16973239, 1497.19062847, 960.16434834, 1278.01658368,
              1542.1350215, 1811.48969781])
y = np.array([0., -277.32014705, 382.62174258, 88.0663819,
              -172.46333123, -418.20411911, 601.89813793, 276.65297749,
              58.93331112, -183.47162896, 689.51974012, 503.04584814,
              283.32467309, -4.33660214])

# Simulation parameters
wake_angle = 20  # Wake expansion angle in degrees
wind_direction = 210  # Wind direction in degrees (meteorological convention)
turbine_diameter = 82  # Turbine diameter in meters
D=turbine_diameter
# Create figure
fig, ax = plt.subplots(figsize=(12, 8))
print(f"Axis limits before setting: {ax.get_xlim()}, {ax.get_ylim()}")
ax.set_xlim(0, 2200)
ax.set_ylim(-500, 800)
print(f"Axis limits after setting: {ax.get_xlim()}, {ax.get_ylim()}")
# Dummy weight function (replace with your actual implementation)
weight = weight_function(x, y, wind_direction, turbine_diameter)

# Plot wake regions
ax = plot_wake_region(ax, x[weight != 0], y[weight != 0], wind_direction, D, wake_angle)

# Plot turbines
ax.plot(x[weight != 0], y[weight != 0], 'bo', label='Included Turbines')
ax.plot(x[weight == 0], y[weight == 0], 'ro', label='Excluded Turbines')

# Add wind direction arrow (converted to Cartesian coordinates)
wind_dir_rad = np.deg2rad(270 - wind_direction)
ax.quiver(5*D,-7.5*D, np.cos(wind_dir_rad), np.sin(wind_dir_rad), 
          scale=10, color='green', label=f'Wind Direction ({wind_direction}°)')

#for i, (xi, yi) in enumerate(zip(x, y)):
 #   plt.text(xi-40, yi, f' {i+1}', va='top', ha='right')
# Configure plot
ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
#ax.set_title(f'Wind Farm Layout with Wake Regions ({wind_direction}° Wind)')
ax.axis('equal')
ax.legend(loc='lower right',fontsize=18)
#ax.grid(True)
#plt.tight_layout()
plt.show()