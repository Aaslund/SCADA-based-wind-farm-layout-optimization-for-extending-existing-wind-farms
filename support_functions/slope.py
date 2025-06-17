# -*- coding: utf-8 -*-
"""
Created on Tue May  6 08:47:12 2025

@author: erica
"""
import numpy as np
from math import atan2, degrees, sqrt

x = np.array([0., 319.22363416, 230.58157057, 500.06092411, 774.34016385, 1030.60424373, 565.70093313, 909.048672, 1172.16973239, 1497.19062847, 960.16434834, 1278.01658368, 1542.1350215, 1811.48969781])
y = np.array([0., -277.32014705, 382.62174258, 88.0663819, -172.46333123, -418.20411911, 601.89813793, 276.65297749, 58.93331112, -183.47162896, 689.51974012, 503.04584814, 283.32467309, -4.33660214])
z = np.array([212.26, 200.46, 208.91, 201.38, 199.03, 180.24, 200.13, 187.04, 186.88, 204.84, 219.31, 220, 219.46, 228.15])

n = len(x)
slopes = []

for i in range(n):
    for j in range(i+1, n):
        dx = x[i] - x[j]
        dy = y[i] - y[j]
        dz = z[i] - z[j]
        horizontal_distance = sqrt(dx**2 + dy**2)
        if horizontal_distance > 0:  # Avoid division by zero
            slope_rad = atan2(abs(dz), horizontal_distance)
            slope_deg = degrees(slope_rad)
            slopes.append(slope_deg)

mean_slope = np.mean(slopes)
median_slope = np.median(slopes)
std_slope = np.std(slopes)

print(f"Mean slope: {mean_slope:.2f}°")
print(f"Median slope: {median_slope:.2f}°")
print(f"Slope standard deviation: {std_slope:.2f}°")
