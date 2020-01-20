#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derive the freestream wind speed or power based on scada measurements of Power 
The reference wind direction is based on a mesoscale virtual time series at the
wind farm centroid
@author: Javier Sanz Rodrigo
"""
import numpy as np
import pandas as pd
from scipy import interpolate
import netCDF4
import matplotlib.pyplot as plt
from time import sleep
from tqdm import tqdm
from scipy import interpolate

import alphashape
from descartes import PolygonPatch
from shapely.geometry import Point, LineString

from WindConditions import *
from BinAvrg import *  

def define_freestream(points,WD):
    """
    Define which turbines in the layout are in freestream conditions
    Inputs:
        - points: array with x,y coordinates [m]
        - WD: wind direction 
    Outputs:
        - freestream: boolean array with freestream turbines = True   
    """
    raylength = 20000. # distance [m] upstream from the centroid to do ray-casting along a line that passes through each turbine in the boundary of the alphashape 
             # make sure the ray is always out of the wind farm
    sectorwidth = 45. # wind direction sector [deg] free of upstream turbines around the input wind direction 
    alpha = 0.0004    # controls the tightness of the bounding shape to the layout (alpha = 0 returns the convex hull)
    Nwt = points.shape[0] # number of turbines
    
    alpha_shape = alphashape.alphashape(points, alpha) # polygon of the wind farm to define turbines in freestream conditions (= turbines in the boundary)
    freestream = np.zeros(Nwt, dtype=bool)
    wd = np.array([WD - sectorwidth/2, WD + sectorwidth/2])
    polarangle = (-wd + 90.)*np.pi/180. # from wind direction to polar angle
    for wt in range(Nwt):
        point_turbine = Point(points[wt,:])
        point_upstream1 = Point(point_turbine.x + raylength*np.cos(polarangle[0]), point_turbine.y + raylength*np.sin(polarangle[0]))
        point_upstream2 = Point(point_turbine.x + raylength*np.cos(polarangle[1]), point_turbine.y + raylength*np.sin(polarangle[1]))
        if alpha_shape.exterior.distance(point_turbine) <= 10.: # turbine in the boundary
            line1 = LineString([point_upstream1, point_turbine])
            line2 = LineString([point_upstream2, point_turbine])
            if (not line1.crosses(alpha_shape.exterior)) and (not line2.crosses(alpha_shape.exterior)):
                freestream[wt] = True
                
    return freestream


siteID = 'Ormonde'

# Load wind farm layout data
turbines = pd.read_csv('../' + siteID + '/inputs/' + siteID + '_layout.csv')
x_ref, y_ref = centroid(turbines[['X coordinate','Y coordinate']].values) # coordinates of wind farm centroid
Nwt = turbines.shape[0]
pwr_curve_file = pd.read_csv('../' + siteID + '/inputs/' + siteID + '_pwc.csv')
pwr_curve_file['Power'] = pwr_curve_file['Power']/1000 # scale to MW
pwr_curve_inv = interpolate.interp1d(pwr_curve_file['Power'].values.flatten(),pwr_curve_file['U'].values.flatten(), 
                                bounds_error = False, fill_value = 0) # inverse of the power curve

# Load scada (power) data
obs_ts = pd.read_csv('../' + siteID + '/observations/' + siteID + '_obs.csv', index_col = 'time') 
#scada_flags = pd.read_csv('../' + siteID + '/inputs/' + siteID + '_flags.csv', index_col = 'Datetime') 

# Load mesoscale data 
try:
    f = netCDF4.Dataset('../' + siteID + '/inputs/' + siteID + '_Wakes_ref.nc', 'r')
    fwt = netCDF4.Dataset('../' + siteID + '/inputs/' + siteID + '_Wakes_WindTurbines.nc', 'r')
except:
    f = netCDF4.Dataset('../' + siteID + '/inputs/' + siteID + '_Control_ref.nc', 'r')
    fwt = netCDF4.Dataset('../' + siteID + '/inputs/' + siteID + '_Control_WindTurbines.nc', 'r')
    
Sref = pd.Series(
            (f.variables['U'][:].data**2 + f.variables['V'][:].data**2)**0.5, 
            index = f.variables['Times'][:].data)
WDref = pd.Series(
        180 + np.arctan2(f.variables['U'][:].data,f.variables['V'][:].data)*180/np.pi,
        index = f.variables['Times'][:].data)

Sref = Sref.reindex(obs_ts.index)
WDref = WDref.reindex(obs_ts.index)

Nobs = obs_ts.shape[0]
points = turbines[['X coordinate','Y coordinate']].values
#
## One freestream for all turbines (homogeneous inflow)
#obs_S_ts = pd.Series(index = obs_ts.index)
#obs_P_ts = pd.Series(index = obs_ts.index)
#pbar = tqdm(total = Nobs, leave = False, position = 0, desc="Progress")
#for t in range(Nobs):
#    WD = WDref.iloc[t]
#    if not np.isnan(WD):
#        freestream = define_freestream(points,WD)
#        Pfree = np.mean(obs_ts.iloc[t,freestream])
#        if Pfree < pwr_curve_file['Power'].max():
#            obs_S_ts.iloc[t] = pwr_curve_inv(Pfree)
#            obs_P_ts.iloc[t] = Pfree
#    pbar.update(1)
#pbar.close()

# Individualized freestream based on front row of turbines (heterogeneous inflow) 
obs_S_ts = pd.DataFrame(np.zeros((Nobs,Nwt)), index = obs_ts.index, columns = turbines['VDC ID'].values)
obs_P_ts = pd.DataFrame(np.zeros((Nobs,Nwt)), index = obs_ts.index, columns = turbines['VDC ID'].values)
pbar = tqdm(total = Nobs, leave = False, position = 0, desc="Progress")
for t in range(Nobs):
    WD = WDref.iloc[t]
    if not np.isnan(WD):
        freestream = define_freestream(points,WD)
        f1 = interpolate.NearestNDInterpolator(points[freestream], obs_ts.iloc[t,freestream])
        Pfree = f1(points)
        obs_S_ts.iloc[t] = pwr_curve_inv(Pfree)
        obs_P_ts.iloc[t] = Pfree.values
    pbar.update(1)
pbar.close()

#fileout = '../' + siteID + '/observations/' + siteID + '_Sobs.csv' 
#obs_S_ts.to_frame().to_csv(fileout)

fileout = '../' + siteID + '/observations/' + siteID + '_Pfree.csv' 
#obs_P_ts.to_csv(fileout)

fileout = '../' + siteID + '/observations/' + siteID + '_Sfree.csv' 
obs_S_ts.to_csv(fileout)

# Load bin-averaged local mesoscale bias correction factors Am
Sbins = np.array([8,10])              # around the maximum of the trust coefficient 
WDbins = np.arange(-15.,360.+15.,30)  # wind direction bins (12 sectors)
WDbins_label = ['N','NNE','ENE','E','ESE','SSE',
                'S','SSW','WSW','W','WNW','NNW']
zLbins = [-0.2,-0.02, 0.02, 0.2]      # 3 stability bins
zLbins_label = ['u','n','s']
Am = xr.open_dataarray('../' + siteID + '/inputs/' + siteID +'_Am.nc')
Nwt, Nwd, NzL = Am.shape

# Plot As for a given bin
WDbin = 'WSW' # select wind direction sector
zLbin = 'n'   # select stability

data = Am.loc[:,WDbin,zLbin].to_pandas()
cmap=plt.cm.get_cmap('jet',8)
fig, ax = plt.subplots(figsize=(10,6))
sc = ax.scatter(turbines['X coordinate']-x_ref,turbines['Y coordinate']-y_ref
                     ,marker='o',c=data,cmap=cmap,edgecolors ='k', vmin = 0.85, vmax=1.1)
plt.colorbar(sc,ax=ax)
ax.set_aspect(1.0)
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title(r'$A_{M,i}$ ('+ WDbin + ',' + zLbin + ')')
ax.grid()

# Apply mean corrections to mesoscale input data


# Generate netcdf input 
#fileout = '../' + siteID + '/input/' + siteID + '_Wakes_WindTurbines_corrected.csv' 


### Test freestream function
#WD = 255.
#freestream = define_freestream(points,WD)
#wt = 9 # execute for loop for this t
#wd = np.array([WD - sectorwidth/2, WD + sectorwidth/2])
#polarangle = (-wd + 90.)*np.pi/180. # from wind direction to polar angle
#point_turbine = Point(points[wt,:])
#point_upstream1 = Point(point_turbine.x + raylength*np.cos(polarangle[0]), point_turbine.y + raylength*np.sin(polarangle[0]))
#point_upstream2 = Point(point_turbine.x + raylength*np.cos(polarangle[1]), point_turbine.y + raylength*np.sin(polarangle[1]))
#line1 = LineString([point_upstream1, point_turbine])
#line2 = LineString([point_upstream2, point_turbine])
#                
#fig, ax = plt.subplots(figsize = (8,8))
#point_upstream1 = Point(point_turbine.x + raylength*np.cos(polarangle), point_turbine.y + raylength*np.sin(polarangle))
#line0 = LineString([point_upstream1, point_turbine]) 
#point_ref = Point(alpha_shape.centroid)
#ax.plot(points[:,0],points[:,1],'o',markerfacecolor = "None")
#ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
#x1,y1 = alpha_shape.exterior.xy
#ax.plot(x1,y1,'r')
#ax.plot(point_ref.x,point_ref.y,'+k', markersize = 20)
#x2,y2 = line1.xy
#ax.plot(x2,y2,'-.c')
#x3,y3 = line2.xy
#ax.plot(x3,y3,'-.c')
#x0,y0 = line0.xy
#ax.plot(x0,y0,'--',color='grey')
#ax.plot(points[wt,0],points[wt,1],'oc', markersize = 9)
#ax.text(points[wt,0]+200,points[wt,1]+200,turbines['VDC ID'][wt], weight='bold')
#ax.text(point_ref.x+200,point_ref.y+200,'ref')
#ax.plot(points[freestream,0],points[freestream,1],'or', markersize = 6)
#ax.set_aspect(1)
#ax.set_title('Wind direction = ' + str(WD))
#ax.set_xlabel('X [m]')
#ax.set_ylabel('X [m]')
#ax.grid()
#
##plt.savefig('freestream.png', dpi=300, bbox_inches='tight')
raylength = 20000. # distance [m] upstream from the centroid to do ray-casting along a line that passes through each turbine in the boundary of the alphashape 
             # make sure the ray is always out of the wind farm
sectorwidth = 45. # wind direction sector [deg] free of upstream turbines around the input wind direction 
alpha = 0.0004    # controls the tightness of the bounding shape to the layout (alpha = 0 returns the convex hull)
alpha_shape = alphashape.alphashape(points, alpha)
    
t = 110
WD = WDref.iloc[t]
S = Sref.iloc[t]
freestream = define_freestream(points,WD)

point_ref = Point(alpha_shape.centroid)

vmin = 4
vmax = 6

data0 = obs_ts.iloc[t,:]
cmap=plt.cm.get_cmap('jet',10)
fig, ax = plt.subplots(1, 2, figsize=(12,6), sharey = True)
sc0 = ax[0].scatter(turbines['X coordinate'],turbines['Y coordinate'],
                     marker='o',c=data0,cmap=cmap,edgecolors ='k', vmin=vmin, vmax=vmax)
#ax[0].add_patch(PolygonPatch(alpha_shape, alpha=0.2))
#x1,y1 = alpha_shape.exterior.xy
#ax[0].plot(x1,y1,'r')
ax[0].plot(point_ref.x,point_ref.y,'+k', markersize = 20)
ax[0].text(point_ref.x+200,point_ref.y+200,'ref')
ax[0].plot(points[freestream,0],points[freestream,1],'or', markersize = 10, markerfacecolor = "None")
#plt.colorbar(sc0,ax=ax[0])
ax[0].set_aspect(1.0)
ax[0].set_xlabel('X [m]')
ax[0].set_ylabel('Y [m]')
ax[0].set_title(r'$P_{obs}$')
ax[0].grid()

data1 = obs_P_ts.iloc[t,:]
sc1 = ax[1].scatter(turbines['X coordinate'],turbines['Y coordinate'],
                     marker='o',c=data1,cmap=cmap,edgecolors ='k', vmin=vmin, vmax=vmax)
#plt.colorbar(sc1,ax=ax[1])
ax[1].set_aspect(1.0)
ax[1].set_xlabel('X [m]')
ax[1].set_title(r'$P_{obsfree}$')
ax[1].grid()

fig.subplots_adjust(right=1.45)
cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
fig.colorbar(sc0, cax=cbar_ax)

fig.suptitle(r'$WD_{t}$ ='+ "{:5.2f}".format(WD) + r'º; $S_{t}$ = ' + "{:5.2f}".format(S) + ' m/s', fontsize=12)
plt.tight_layout()





