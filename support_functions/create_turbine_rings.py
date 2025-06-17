# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 13:17:19 2025

@author: erica
"""

import numpy as np

def create_turbine_rings(x, y, min_spacing, num_points=50):
    rings = []
    
    for xi, yi in zip(x, y):
        # --- First half-donut (0° to 180°) ---
        angles1 = np.linspace(0, 2*np.pi, num_points, endpoint=True)  # Closed (includes endpoint)
        
        # Outer semicircle (counter-clockwise)
        x_outer = xi + min_spacing * np.cos(angles1)
        y_outer = yi + min_spacing * np.sin(angles1)
        
        # Combine and close
        ring = np.column_stack([x_outer,y_outer])
        
        rings.append(ring)
    
    return rings