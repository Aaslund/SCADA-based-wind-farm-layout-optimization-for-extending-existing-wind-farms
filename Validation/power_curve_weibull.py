# -*- coding: utf-8 -*-
"""
Created on Tue Jun  3 16:14:10 2025

@author: erica
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import weibull_min

# Given power curve data
u = np.array([3.5, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25.1, 30])
power = np.array([0, 64, 159, 314, 511, 767, 1096, 1439, 1700, 1912, 2000, 2040, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 2050, 0, 0])

# Create 0.5 m/s bins from 3.5 to 25.0 m/s
u_bins = np.arange(3.5, 25.5, 0.5)
bin_centers = (u_bins[:-1] + u_bins[1:]) / 2
bin_widths = np.diff(u_bins)

# Interpolate the power curve to the new bin centers
power_interp = np.interp(bin_centers, u, power)

k = 1.9
A = 8.97

# Evaluate Weibull PDF at new bin centers
ref = weibull_min.pdf(bin_centers, k, scale=A)

# Calculate energy contribution per bin (probability * power)
probability_per_bin = ref * bin_widths
hours_per_year = 8760
energy_contribution_kwh = 14*power_interp * probability_per_bin * hours_per_year / 1e6  # Convert from Wh to kWh

# Fix the legend collection from both axes

# Create plot with secondary y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))

# Left y-axis: power curve and Weibull distribution
l1, = ax1.plot(u, power, 'b-', label='Power Curve')
l2, = ax1.plot(bin_centers, ref * max(power)/max(ref), 'g--', label='Weibull Distribution (scaled)')
ax1.set_xlabel('Wind Speed (m/s)')
ax1.set_ylabel('Power (W)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
vline = ax1.vlines(14.5, 0, 2100, linestyles='--', colors='grey', label='Rated Wind Speed')

# Right y-axis: energy contribution
ax2 = ax1.twinx()
ax2.bar(bin_centers, energy_contribution_kwh, color='r', label='AEP', alpha=0.5, width=0.4)
ax2.set_ylabel('Energy Contribution (MWh)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Combine legends
#lines = [l1, l2, l3, vline]
#labels = [line.get_label() for line in lines]
#ax1.legend(lines, labels, loc='upper right')

plt.tight_layout()
plt.show()
