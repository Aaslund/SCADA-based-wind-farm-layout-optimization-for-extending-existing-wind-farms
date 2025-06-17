# -*- coding: utf-8 -*-
"""
Created on Tue May 27 19:12:47 2025

@author: erica
"""

from topfarm.plotting import XYPlotComp
import numpy as np
import topfarm

class CustomXYPlotComp(XYPlotComp):
    def __init__(self, diameter=82, existing_x=None, existing_y=None, xlim=None, ylim=None, xlabel='x [m]', ylabel='y [m]', xticks=None, yticks=None, fontsizelegend=12, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._diameter = diameter
        self._custom_xlim = xlim
        self._custom_ylim = ylim
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._xticks = xticks
        self._yticks = yticks
        self.existing_x = np.asarray(existing_x) if existing_x is not None else None
        self.existing_y = np.asarray(existing_y) if existing_y is not None else None
        self.fontsizelegend = fontsizelegend
    # def _normalize(self, arr):
    #     return np.asarray(arr) / self._diameter if arr is not None else None

    def set_title(self, cost0, cost):
        rec = self.problem.recorder
        iteration = rec.num_cases
        # Get the increase/expansion factor if it exists, otherwise default to 1
        inc_or_exp = getattr(self.problem.cost_comp, 'inc_or_exp', 1.0)

        # Compute percentage delta safely
        delta = ((cost - cost0) / cost0 * 100) if cost0 else 0
        if np.sign(delta) == -1:
            sign=''
        else:
            sign='+'
        # Create a clean and informative title
        title = f"Iterations: {iteration} | AEP: {cost*1000 * inc_or_exp:.2f} MWh  |  Δ: {sign}{delta:.1f}%"

        # Set the title with custom formatting
        self.ax.set_title(title, fontsize=15)

    def init_plot(self, limits):
        super().init_plot(limits) 

        if self._custom_xlim and self._custom_ylim:
            self.ax.set_xlim(*self._custom_xlim)
            self.ax.set_ylim(*self._custom_ylim)
        else:
            mi = limits.min(0)
            ma = limits.max(0)
            ra = ma - mi + 1
            ext = .1
            xlim, ylim = np.array([mi - ext * ra, ma + ext * ra]).T
            self.ax.set_xlim(xlim)
            self.ax.set_ylim(ylim)
        # Set axis labels
        self.ax.set_xlabel(self._xlabel)
        self.ax.set_ylabel(self._ylabel)

        # Set custom ticks if provided
        if self._xticks is not None:
            self.ax.set_xticks(self._xticks)
        if self._yticks is not None:
            self.ax.set_yticks(self._yticks)
        
            
    def plot_existing_turbines(self):
        # x_norm = self._normalize(self.existing_x)
        # y_norm = self._normalize(self.existing_y)
        if self.existing_x is not None and self.existing_y is not None:
            for x, y in zip(self.existing_x, self.existing_y):
                self.ax.plot(x, y, 'o', markerfacecolor='grey', markeredgecolor='black', markersize=6)
            # For legend (only adds once)
            self.ax.plot([], [], 'o', markerfacecolor='grey', markeredgecolor='black', label='Existing turbines')

    def compute(self, inputs, outputs):
        # inputs_copy = {key: inputs[key] for key in inputs.keys()}
        # inputs_copy[topfarm.x_key] = self._normalize(inputs[topfarm.x_key])
        # inputs_copy[topfarm.y_key] = self._normalize(inputs[topfarm.y_key])
        # if self.cost_key in inputs:
        #     inputs_copy[self.cost_key] = inputs[self.cost_key]

        # Save original inputs and call base compute
        #super().compute(inputs_copy, outputs)
        super().compute(inputs, outputs)
        self.plot_existing_turbines()
        # Then adjust legend
        handles, labels = self.ax.get_legend_handles_labels()
        self.ax.legend(handles, labels, loc='lower right', fontsize=self.fontsizelegend)
    def save_fig(self, filepath, dpi=300):
        self.ax.figure.savefig(filepath, dpi=dpi)