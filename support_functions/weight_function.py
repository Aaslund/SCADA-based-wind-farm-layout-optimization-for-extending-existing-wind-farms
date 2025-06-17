import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import pandas as pd
#from haversine import haversine
plt.rcParams.update({'font.size': 20})
def weight_function(x, y, wind_direction, turbine_diameter, wake_angle=20):
    """
    Calculate weights for turbines based on wake effects, considering the entire rotor width.

    Parameters:
        x (np.array): X positions of turbines
        y (np.array): Y positions of turbines
        wind_direction (float): Wind direction in degrees (meteorological convention)
        turbine_diameter (float): Diameter of the wind turbine
        wake_angle (float): Wake expansion angle in degrees (default: 5°)

    Returns:
        np.array: Weights for each turbine (1 = unaffected, 0 = in wake)
    """
    num_turbines = len(x)
    weights = np.ones(num_turbines)  # All turbines start unaffected

    # Convert wind direction to radians (FIX: Rotate by -90° to match Cartesian system)
    wind_dir_rad = np.deg2rad(270 - wind_direction)  # 270-wind_direction converts to Cartesian
    
    # Wind vector (normalized)
    wind_vector = np.array([np.cos(wind_dir_rad), np.sin(wind_dir_rad)])

    for i in range(num_turbines):
        for j in range(num_turbines):
            if i == j:
                continue  # Skip self-check

            # Compute displacement from turbine j to turbine i
            dx, dy = x[i] - x[j], y[i] - y[j]
            distance = np.sqrt(dx**2 + dy**2)
            if distance == 0:
                continue

            # Compute perpendicular distance from wake centerline
            cross_product = dx * wind_vector[1] - dy * wind_vector[0]
            # **Wake Region Definition**
            projection = np.dot([dx, dy], wind_vector)  # Projection of turbine i onto wind vector (to check if it's downwind)
            if projection > 0:
                lateral_distance = abs(cross_product)  # Distance from wake axis
                if lateral_distance < turbine_diameter / 2 + projection * np.tan(np.deg2rad(wake_angle)):  # Turbine j is upwind of turbine i
                    weights[i] = 0  # Mark as affected by wake

    return weights

# def weight_function_multiple_directions(x, y, wind_directions, turbine_diameter, wake_angle=5):
#     """
#     Calculate weights for turbines for multiple wind directions by calling the original weight_function.

#     Parameters:
#         x (np.array): X positions of turbines
#         y (np.array): Y positions of turbines
#         wind_directions (list): List of wind directions in degrees (meteorological convention)
#         turbine_diameter (float): Diameter of the wind turbine
#         wake_angle (float): Wake expansion angle in degrees (default: 5°)

#     Returns:
#         dict: Dictionary of weights for each wind direction (1 = unaffected, 0 = in wake)
#     """
#     weights = {}
#     for wind_direction in wind_directions:
#         weights[wind_direction] = weight_function(x, y, wind_direction, turbine_diameter, wake_angle)
#     return weights

# def plot_wake_region(ax, x, y, wind_direction, turbine_diameter, wake_angle=5):
#     """
#     Plot the wake region as a light blue polygon behind the turbines for a single wind direction.
#     """
#     # Convert wind direction to radians (adjust for Cartesian coordinates)
#     wind_dir_rad = np.deg2rad(270-wind_direction)
#     wind_vector = np.array([np.cos(wind_dir_rad), np.sin(wind_dir_rad)])

#     # Distance to extend the wake region
#     wake_length = 11  # Extend wake to plot limits

#     # Calculate the wake region for each turbine
#     for turbine_x, turbine_y in zip(x, y):
#         # Calculate the wake edges
#         # Left and right edges at the turbine position
#         left_edge = [
#             turbine_x - (turbine_diameter / 2) * wind_vector[1],
#             turbine_y + (turbine_diameter / 2) * wind_vector[0],
#         ]
#         right_edge = [
#             turbine_x + (turbine_diameter / 2) * wind_vector[1],
#             turbine_y - (turbine_diameter / 2) * wind_vector[0],
#         ]

#         # Left and right edges at the end of the wake region
#         left_edge_end = [
#             left_edge[0] + wake_length * wind_vector[0] - wake_length * np.tan(np.deg2rad(wake_angle)) * wind_vector[1],
#             left_edge[1] + wake_length * wind_vector[1] + wake_length * np.tan(np.deg2rad(wake_angle)) * wind_vector[0],
#         ]
#         right_edge_end = [
#             right_edge[0] + wake_length * wind_vector[0] + wake_length * np.tan(np.deg2rad(wake_angle)) * wind_vector[1],
#             right_edge[1] + wake_length * wind_vector[1] - wake_length * np.tan(np.deg2rad(wake_angle)) * wind_vector[0],
#         ]

#         # Define the polygon vertices for the wake region
#         wake_polygon = np.array([
#             left_edge,  # Start at left edge of turbine
#             right_edge,  # Move to right edge of turbine
#             right_edge_end,  # Extend to right edge of wake
#             left_edge_end,  # Extend to left edge of wake
#         ])

#         # Create a Polygon object
#         hull = Polygon(wake_polygon, closed=True, color='lightblue', alpha=0.3)

#         # Add the polygon to the plot
#         ax.add_patch(hull)

# x = np.array([0., 319.22363416, 230.58157057, 500.06092411,
#               774.34016385, 1030.60424373, 565.70093313, 909.048672,
#               1172.16973239, 1497.19062847, 960.16434834, 1278.01658368,
#               1542.1350215, 1811.48969781])
# y = np.array([0., -277.32014705, 382.62174258, 88.0663819,
#               -172.46333123, -418.20411911, 601.89813793, 276.65297749,
#               58.93331112, -183.47162896, 689.51974012, 503.04584814,
#               283.32467309, -4.33660214])
# turbine_diameter = 82  
# wake_angle = 20  # Wake expansion angle in degrees
# #%%
# # Define the number of wind directions and generate the list
# plt.rcParams.update({'font.size': 22})
# n = 4
# wind_directions = np.linspace(0, 360, n, endpoint=False)

# # Calculate weights for all wind directions
# weights = weight_function_multiple_directions(x, y, wind_directions, turbine_diameter, wake_angle)

# # Plotting
# fig, axs = plt.subplots(2, 2, figsize=(16, 12))  # Adjust subplot grid based on n
# axs = axs.ravel()

# for ax, wind_direction in zip(axs, wind_directions):
#     weight = weights[wind_direction]

#     # Plot wake region
#     plot_wake_region(ax, x[weight != 0]/turbine_diameter, y[weight != 0]/turbine_diameter, wind_direction, 1, wake_angle)

#     # Plot turbines
#     ax.plot(x[weight != 0]/turbine_diameter, y[weight != 0]/turbine_diameter, 'bo', label='Included Turbines',markersize=8)  # Blue for kept turbines
#     ax.plot(x[weight == 0]/turbine_diameter, y[weight == 0]/turbine_diameter, 'ro', label='Excluded Turbines',markersize=8)  # Red for thrown away turbines

#     # Add wind direction arrow
#     wind_dir_rad = np.deg2rad(270-wind_direction)
#     ax.quiver(10, 2.5, np.cos(wind_dir_rad), np.sin(wind_dir_rad), scale=10, color='green', label='Wind Direction')

#     # Set plot limits and labels
#     ax.set_xlim(0, 26)
#     ax.set_ylim(-10, 10)
#     ax.set_xlabel('x/D [-]')
#     ax.set_ylabel('y/D [-]')
#     ax.set_title(f'{wind_direction:.0f}° Wind')
#     ax.axis('equal')
#     ax.legend(fontsize='16',loc='upper right')

# fig.tight_layout()
# plt.show()