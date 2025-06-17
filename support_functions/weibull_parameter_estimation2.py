# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 11:46:42 2025

@author: erica
"""
import numpy as np
from scipy.stats import weibull_min, kstest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def weibull_pdf2(x, k, A):
    return (k / A) * (x / A) ** (k - 1) * np.exp(- (x / A) ** k)

def estimate_weibull_mle2(wind_speeds):
    k, loc, A = weibull_min.fit(wind_speeds, floc=0)
    # k is the shape parameter
    # A is the scale parameter
    return k, loc, A

# === 5. Performance Evaluation ===
def evaluate_weibull_fit2(wind_speeds, k, loc, A, season, wind_dir_i, bin_width=0.5, rated_wind_speed=14.5):
    """
    Evaluate Weibull fit with fixed bin width instead of fixed number of bins
    
    Parameters:
        wind_speeds (array): Array of wind speed measurements
        k (float): Weibull shape parameter
        loc (float): Weibull location parameter
        A (float): Weibull scale parameter
        season (str): Season identifier
        wind_dir_i (float): Wind direction in degrees
        bin_width (float): Width of bins in m/s (default: 0.5)
    """
    # Calculate number of bins based on desired width
    max_speed = max(wind_speeds)
    
    # Create bins from 0 to max_speed+bin_width with specified width
    bin_edges = np.arange(0, max_speed + bin_width, bin_width)
    hist, bin_edges = np.histogram(wind_speeds, bins=bin_edges, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pdf_weibull = weibull_pdf2(bin_centers, k, A)

    # Calculate statistics
    rmse = np.sqrt(mean_squared_error(hist, pdf_weibull))
    mae = mean_absolute_error(hist, pdf_weibull)
    r2 = r2_score(hist, pdf_weibull)    
    ks_stat, p_value = kstest(wind_speeds, 'weibull_min', args=(k, loc, A))
    correlation = np.corrcoef(hist, pdf_weibull)[0, 1]
    
    mask_above_rated = bin_centers > rated_wind_speed
    if np.any(mask_above_rated):
        hist_above = hist[mask_above_rated]
        pdf_above = pdf_weibull[mask_above_rated]
        error_above_rated = np.abs(np.sum((hist_above - pdf_above) * bin_width))
    else:
        error_above_rated = np.nan
    
    
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Kolmogorov-Smirnov Statistic: {ks_stat:.4f}")
    print(f"Kolmogorov-Smirnov p-value: {p_value:.4f}")
    print(f"Correlation Coefficient: {correlation:.4f}\n")
    print(f"Tail Error: {error_above_rated:.4f}\n")
    
    return ks_stat, p_value, rmse, error_above_rated

