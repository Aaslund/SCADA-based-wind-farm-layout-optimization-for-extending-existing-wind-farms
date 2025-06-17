# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 17:57:20 2025

@author: erica
"""
import numpy as np

def expand_param(param, nturb):
    if isinstance(param, (list, np.ndarray)):
        return np.array(param)
    else:
        return np.full(nturb, param)