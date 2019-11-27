#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Derive the freestream wind speed based on scada measurements of Power 
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

import alphashape
from descartes import PolygonPatch
from shapely.geometry import Point, LineString


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
    alpha = 0.0004    # controls the tightness of the bounding shape to the layour (alpha = 0 returns the convex hull)
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
pwr_curve_file = pd.read_csv('../' + siteID + '/inputs/' + siteID + '_pwc.csv')
pwr_curve_file['Power'] = pwr_curve_file['Power']/1000 # scale to MW
pwr_curve_inv = interpolate.interp1d(pwr_curve_file['Power'].values.flatten(),pwr_curve_file['U'].values.flatten(), 
                                bounds_error = False, fill_value = 0) # inverse of the power curve

# Load scada data
obs_ts = pd.read_csv('../' + siteID + '/observations/' + siteID + '_obs.csv', index_col = 'time') 
#scada_flags = pd.read_csv('../' + siteID + '/inputs/' + siteID + '_flags.csv', index_col = 'Datetime') 

# Load mesoscale ref data 
try:
    f = netCDF4.Dataset('../' + siteID + '/inputs/' + siteID + '_Wakes_ref.nc', 'r')
except:
    f = netCDF4.Dataset('../' + siteID + '/inputs/' + siteID + '_Control_ref.nc', 'r')
    
Sref = pd.Series(
            (f.variables['U'][:].data**2 + f.variables['V'][:].data**2)**0.5, 
            index = f.variables['Times'][:].data)
WDref = pd.Series(
        180 + np.arctan2(f.variables['U'][:].data,f.variables['V'][:].data)*180/np.pi,
        index = f.variables['Times'][:].data)

Sref = Sref.reindex(obs_ts.index)
WDref = WDref.reindex(obs_ts.index)

obs_S_ts = pd.Series(index = obs_ts.index)
obs_P_ts = pd.Series(index = obs_ts.index)

Nobs = obs_ts.shape[0]
points = turbines[['X coordinate','Y coordinate']].values

pbar = tqdm(total = Nobs, leave = False, position = 0, desc="Progress")
for t in range(Nobs):
    WD = WDref.iloc[t]
    if not np.isnan(WD):
        freestream = define_freestream(points,WD)
        Pfree = np.mean(obs_ts.iloc[t,freestream])
        if Pfree < pwr_curve_file['Power'].max():
            obs_S_ts.iloc[t] = pwr_curve_inv(Pfree)
            obs_P_ts.iloc[t] = Pfree
    pbar.update(1)
pbar.close()
    
ax = obs_S_ts.iloc[200:500].plot(color = 'b', label = 'obs')    
Sref.iloc[200:500].plot(ax = ax, color = 'r', label = 'meso')    
ax.legend();    

#fileout = '../' + siteID + '/observations/' + siteID + '_Sobs_mean.csv' 
#obs_S_ts.to_frame().to_csv(fileout)

fileout = '../' + siteID + '/observations/' + siteID + '_Pfree_mean.csv' 
obs_P_ts.to_frame().to_csv(fileout)

## Test freestream function
#WD = 325.
#freestream = define_freestream(points,WD)
#wt = 9 # execute for loop for this t
#                
#fig, ax = plt.subplots(figsize = (8,8))
#polarangle = (-WD + 90.)*np.pi/180
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
#ax.plot(points[t,0],points[t,1],'oc', markersize = 9)
#ax.text(points[t,0]+200,points[t,1]+200,turbines['VDC ID'][t], weight='bold')
#ax.text(point_ref.x+200,point_ref.y+200,'ref')
#ax.plot(points[freestream,0],points[freestream,1],'or', markersize = 6)
#ax.set_aspect(1)
#ax.set_title('Wind direction = ' + str(WD))
#ax.set_xlabel('X [m]')
#ax.set_ylabel('X [m]')
#ax.grid()

#plt.savefig('freestream.png', dpi=300, bbox_inches='tight')

