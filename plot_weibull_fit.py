# -*- coding: utf-8 -*-
"""
Created on Sat May 31 11:30:12 2025

@author: erica
"""
import numpy as np
from scipy.stats import weibull_min, kstest
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

def plot_weibull_fit(df, k, A, season, wind_dir_i, font_size=15, figsize=(10,6), save_path=None, xlimit=None, ylimit=None):
    bin_width = 0.5
    hist, bin_edges = np.histogram(df['wind_speed'], bins=np.arange(0, 25+bin_width, bin_width), density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    pdf_weibull = weibull_min.pdf(bin_centers, k, scale=A)
    ref = weibull_min.pdf(bin_centers, 1.9, scale=8.97)
    new = weibull_min.pdf(bin_centers, 2.13, scale=10.17)
    # Plot
    plt.figure(figsize=figsize)

    bar_color = '#1f77b4'
    line_color = '#d62728'
    background_color = '#f8f8f8'
    
    plt.bar(bin_centers, hist, width=bin_width*0.9, color=bar_color, edgecolor='white', linewidth=0.5,
            label=f"Empirical PDF")#" ({season}, {round(wind_dir_i,1)}°)")
    plt.plot(bin_centers, pdf_weibull, color=line_color, linewidth=2.5,
             label=f"Weibull Fit (k: {round(k,2)}, A: {round(A,2)})")
    # plt.bar(bin_centers, hist, width=bin_width*0.9, color=bar_color, edgecolor='white', linewidth=0.5,
    #         label="Empirical PDF")
    # plt.plot(bin_centers, pdf_weibull, color=line_color, linewidth=2.5,
    #          label="Weibull Fit")
    # plt.plot(bin_centers, ref, color=line_color, linewidth=2.5,alpha=0.3,linestyle='--',
             # label="Reference (k: 1.90, A: 8.97)")
    plt.plot(bin_centers, new, color='g', linewidth=2.5,
             label="Alternative Weibull Fit (k: 2.13, A: 10.17)")

    plt.gca().set_facecolor(background_color)
    plt.xlabel("Wind Speed [m/s]", fontsize=font_size)
    plt.ylabel("Probability Density [-]", fontsize=font_size)
    plt.xticks(np.arange(0, 25, 5), fontsize=font_size)
    plt.yticks([0.00, 0.05, 0.10, 0.15, 0.2], fontsize=font_size)
    if xlimit is not None:
        plt.xlim(xlimit[0],xlimit[1])
        plt.xticks([15, 17, 19, 21, 23, 25], fontsize=font_size)
    else:
        plt.xlim(0, 25)
    if ylimit is not None:
        plt.ylim(ylimit[0],ylimit[1])
        plt.yticks([0.00, 0.01, 0.02, 0.03, 0.04], fontsize=font_size)
    else:
        plt.ylim(0, 0.20)

    legend = plt.legend(loc='upper right', framealpha=1, edgecolor='black', fontsize=font_size*0.8)
    legend.get_frame().set_facecolor('white')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()