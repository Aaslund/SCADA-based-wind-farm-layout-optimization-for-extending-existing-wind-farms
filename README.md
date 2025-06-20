# SCADA-based-wind-farm-layout-optimization-for-extending-existing-wind-farms

The three main scripts for the master's thesis are 1_importSCADA.py, 2_weibull_fit.py and 3_optimizeExtension.py, that are to be run sequentially.
In addition to these codes that consitute the core of the methodology there are three folders; Plot, support_functions and Validation.

The Plot folder groups the plotting codes that were used to generate the plots in the report.

The support_functions folder groups the functions that are called in the methodology for performing specific tasks, like for instance the masking function, the weibull parameter estimation.

Finally the Validation folder groups all of the scripts that were used to validate the methodology.

The SCADA dataset that was used to design the methodology can be found here: https://zenodo.org/records/5946808

Furthermore the methodology uses the open-source package TOPFARM (https://topfarm.pages.windenergy.dtu.dk/TopFarm2/) for wind farm layout optimization (WFLO) that includes PyWake (https://topfarm.pages.windenergy.dtu.dk/PyWake/) for wind farm modelling and annuel energy production (AEP) calculation.
