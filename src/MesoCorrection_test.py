#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test mesoscale corrections

@author: Javier Sanz Rodrigo
"""

import numpy as np
import pandas as pd
import netCDF4
import matplotlib.pyplot as plt

siteID = 'WestermostRough'

f = netCDF4.Dataset('../' + siteID + '/inputs/' + siteID + '_Wakes_ref_corrected.nc', 'r')
fwt = netCDF4.Dataset('../' + siteID + '/inputs/' + siteID + '_Wakes_WindTurbines_corrected.nc', 'r')

Sref = pd.Series(
            (f.variables['U'][:].data**2 + f.variables['V'][:].data**2)**0.5, 
            index = f.variables['Times'][:].data)
WDref = pd.Series(
        180 + np.arctan2(f.variables['U'][:].data,f.variables['V'][:].data)*180/np.pi,
        index = f.variables['Times'][:].data)

S = pd.DataFrame(
            (fwt.variables['U'][:].data**2 + fwt.variables['V'][:].data**2)**0.5, 
            index = f.variables['Times'][:].data)

Scorr = pd.DataFrame(
            (fwt.variables['U_corr'][:].data**2 + fwt.variables['V_corr'][:].data**2)**0.5, 
            index = f.variables['Times'][:].data)

#Sscada = pd.DataFrame(
#            (fwt.variables['U_scada'][:].data**2 + fwt.variables['V_scada'][:].data**2)**0.5, 
#            index = f.variables['Times'][:].data)

fig, ax = plt.subplots(figsize=(12,6))
ax = S[0].iloc[:200].plot()
Scorr[0].iloc[:200].plot(ax=ax)
#Sscada[0].iloc[:200].plot(ax=ax)
ax.legend(['meso','mesocorr','scada'])
ax.grid()

print('Mean of S at a turbine in the range [8 10]:')
print('Smeso = ' + "{:.2f}".format(S[(S[0]>=8.) & (S[0]<=10)][0].mean()))
print('Scorr = ' + "{:.2f}".format(Scorr[(S[0]>=8.) & (S[0]<=10)][0].mean()))
#print('Sscada = ' + "{:.2f}".format(Sscada[(S[0]>=8.) & (S[0]<=10)][0].mean()))

# number of nans
print('Number on NaNs:')
print('Nmeso = ' + "{:.0f}".format(S[(S[0]>=8.) & (S[0]<=10)][0].isna().sum()) + '/' + "{:.0f}".format(len(S[(S[0]>=8.) & (S[0]<=10)][0])))
print('Ncorr = ' + "{:.0f}".format(Scorr[(S[0]>=8.) & (S[0]<=10)][0].isna().sum()) + '/' + "{:.0f}".format(len(Scorr[(S[0]>=8.) & (S[0]<=10)][0])))
#print('Nscada = ' + "{:.0f}".format(Sscada[(S[0]>=8.) & (S[0]<=10)][0].isna().sum()) + '/' + "{:.0f}".format(len(Sscada[(S[0]>=8.) & (S[0]<=10)][0])))

