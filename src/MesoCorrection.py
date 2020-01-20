# -*- coding: utf-8 -*-
"""
WRFout files to site time-series with tendencies

@authors: Javier Sanz Rodrigo (jsrodrigo@cener.com)
        : Pedro Correia (pmfernandez@cener.com)
"""
# import parameters from file
import sys
sys.path.append("./lib")

from constants_and_functions import *
from datetime import timedelta
import matplotlib.pyplot as plt

site = 'Ormonde'

file_wind_turbines = "../" + site + "/inputs/" + site + "_Wakes_WindTurbines.nc"
file_ref = "../" + site + "/inputs/" + site + "_Wakes_ref.nc"
file_correction = "../" + site + "/inputs/" + site + "_Am.nc"

file_scada = "../" + site + "/observations/" + site + "_Sfree.csv"

outputFile_wind_turbines = "../" + site + "/inputs/" + site + "_Wakes_WindTurbines_corrected.nc"
outputFile_ref = "../" + site + "/inputs/" + site + "_Wakes_ref_corrected.nc"

num_sectores = 12
sec_Labels =['NNE','ENE','E','ESE','SSE','S','SSW','WSW','W','WNW','NNW',"N"]

# Initialize netcdf class
jsonFilesPath='./jsonFiles/'
ncClass = Variables(jsonFilesPath)

f1 = Dataset(file_wind_turbines)
f2 = Dataset(file_ref)
f3 = Dataset(file_correction)

#Dimensions from wind turbine files
nturb = extract_dim(f1, 'turbines')
timesSim = f1.variables['Times'][:]
tdim = len(timesSim)
latTurb = f1.variables["latTurb"][:]
lonTurb = f1.variables["lonTurb"][:]
T2  = f1.variables["T2"][:,:]
TSK = f1.variables["TSK"][:,:]
U10 = f1.variables["U10"][:,:]
V10 = f1.variables["V10"][:,:]
L   = f1.variables["L"][:,:]
UST = f1.variables["UST"][:,:]
HFX = f1.variables["HFX"][:,:]
U   = f1.variables["U"][:,:]
V   = f1.variables["V"][:,:]
POT = f1.variables["POT"][:,:]
TKE = f1.variables["TKE"][:,:]

# L has some NaN values while this is good data. Let's compute L from the fluxes
Cp_air = 1005   # Specific heat of air [J kg-1 K-1]
R_air = 287.058  # Specific gas constant for dry air [J kg-1 K-1]
K  = 0.41       # von Karman constant
g = 9.81    # [m s-2]

#rho = Psfc/(R_air*T2) # didn't save Psfc, let's use the standard density at sea level
rho = 1.225
wt = HFX/(rho*Cp_air)
beta = 1./T2
L0 = -UST**3/(K*g*beta*wt)   # Surface-layer Obukhov length [m]

# Verify that L = L0        
#fig, ax = plt.subplots(figsize=(12,6))
#ax.plot(10./L[0:200,0],'-r', label = 'L')
#ax.plot(10./L0[0:200,0],'-b', label = 'L0')
#ax.legend()
#ax.grid()

L = L0

# File Correction
turb_id = f3.variables["wt"][:]
wd = f3.variables["wd"][:]
zL_lab = f3.variables["zL"][:]
factor = f3.variables["__xarray_dataarray_variable__"][:,:,:]

# Read scada file:
scada = pd.read_csv(file_scada, sep=",")
scada.set_index("time", inplace=True)

# Bining
# Direction
secInt = 360 / num_sectores
centeredBins = np.arange(0 + secInt / 2, 360 + secInt / 2 + secInt, secInt)

# Stability
zLBins = [-0.2, -0.02, 0.02, 0.2]

tsU_corr_df = pd.DataFrame()
tsV_corr_df = pd.DataFrame()

tsU_scada_df = pd.DataFrame()
tsV_scada_df = pd.DataFrame()

for i, turb in enumerate(turb_id):

    wind_dir_Hub = wind_uv_to_dir(U[:,i], V[:,i])
    wind_speed_Hub = uv2Speed(U[:,i], V[:,i])

    factor_turb = pd.DataFrame(factor[i, :, :], columns=[zL_lab])
    factor_turb.index = wd
    factor_turb = factor_turb.unstack().reset_index()
    factor_turb.columns = ['zLBins', 'sector', 'factor']

    # Stability
    zL = 10 / L[:, i].data

    scada_turb = scada[turb]

    # Perform the Binning:
    # To properly group the directions in the "N" sector, 0-5 degrees will become 360-365
    # When direction is between 0 and 5, it add 360, then it subtracts again in the end of the calculation
    # So, for example when having 8 sectors, we will have:
    #  Sec |    limit      |
    #   5  |   [5 to 15[   |
    #   15 |   [15 to 25[  |
    #           ...
    #  355 |  [355 to 365[ |  -> North sector, centered in 0

    wind_turb_df = pd.DataFrame(np.column_stack(
        (timesSim, wind_speed_Hub, wind_dir_Hub, zL)
    ), columns=["Date", "wind_speed_hub", "dir", "zL"])

    wind_turb_df.loc[wind_turb_df['dir'] < centeredBins[0], 'dir'] += 360
    wind_turb_df['DirBins'] = pd.cut(wind_turb_df['dir'], centeredBins, labels=sec_Labels)
    wind_turb_df.loc[wind_turb_df['dir'] >= 360, 'dir'] -= 360

    # Binning the stability
    wind_turb_df['zLBins'] = pd.cut(wind_turb_df['zL'], zLBins, labels=zL_lab)

    # Obtain factor and merge the scada data
    sinc = pd.merge(wind_turb_df, factor_turb,  how='left', left_on=['DirBins','zLBins'], right_on=['sector','zLBins'])
    sinc = pd.merge(sinc, scada_turb,  how='left', left_on="Date", right_index=True)

    # Calculate the new corrected wind speed
    sinc['wind_speed_corrected'] = sinc['wind_speed_hub'] * sinc['factor'] ** (1/3)

    # Obtain the horizontal components using the wind direction at hub height
    u_corr, v_corr = wind_spddir_to_uv(sinc['wind_speed_corrected'].values, wind_dir_Hub.data)
    u_scada, v_scada = wind_spddir_to_uv(sinc[turb].values, wind_dir_Hub.data)

    tsU_corr_df['u_'+turb] = u_corr
    tsV_corr_df['v_'+turb] = v_corr

    tsU_scada_df['u_'+turb] = u_scada
    tsV_scada_df['v_'+turb] = v_scada

# Write to netcdf wind turbines file
f = Dataset(outputFile_wind_turbines , 'w', format="NETCDF4")

f.description = "WRF 3.8.1 - NEWA setup simulation: MYNN2.5 PBL: 27-9-3km 61 sigma levels. Time Series in wind turbines positions."

f.createDimension('time', None)
f.createDimension('turbines', nturb)

# times
secs = ncClass.nc_create(f, 'Times', ('time',))
secs.start_time = f1.variables['Times'].start_time
secs.end_time = f1.variables['Times'].end_time
secs[:] = timesSim

# latTurbines
latTurb_nc = ncClass.nc_create(f, 'latTurb', ('turbines'))
latTurb_nc[:] = latTurb

# lonTurbines
lonturb_nc = ncClass.nc_create(f, 'lonTurb', ('turbines'))
lonturb_nc[:] = lonTurb

# Wind Components at 10m
U10_nc = ncClass.nc_create(f, 'U10', ('time', 'turbines'))
U10_nc[:] = U10

V10_nc = ncClass.nc_create(f, 'V10', ('time', 'turbines'))
V10_nc[:] = V10

# Wind Components at hub height
U_nc = ncClass.nc_create(f, 'U', ('time', 'turbines'))
U_nc[:] = U

V_nc = ncClass.nc_create(f, 'V', ('time', 'turbines'))
V_nc[:] = V

# Wind Components at hub height
U_corr_nc = ncClass.nc_create(f, 'U_corr', ('time', 'turbines'))
U_corr_nc[:] = tsU_corr_df.values

V_corr_nc = ncClass.nc_create(f, 'V_corr', ('time', 'turbines'))
V_corr_nc[:] = tsV_corr_df.values

# Wind Components at hub height
U_scada_nc = ncClass.nc_create(f, 'U_scada', ('time', 'turbines'))
U_scada_nc[:] = tsU_scada_df.values

V_scada_nc = ncClass.nc_create(f, 'V_scada', ('time', 'turbines'))
V_scada_nc[:] = tsV_scada_df.values

T2_nc = ncClass.nc_create(f, 'T2', ('time', 'turbines'))
T2_nc[:] = T2

TSK_nc = ncClass.nc_create(f, 'TSK', ('time', 'turbines'))
TSK_nc[:] = TSK

L_nc = ncClass.nc_create(f, 'L', ('time', 'turbines'))
L_nc[:] = L

UST_nc = ncClass.nc_create(f, 'UST', ('time', 'turbines'))
UST_nc[:] = UST

HFX_nc = ncClass.nc_create(f, 'HFX', ('time', 'turbines'))
HFX_nc[:] = HFX

POT_nc = ncClass.nc_create(f, 'POT', ('time', 'turbines'))
POT_nc[:] = POT

TKE_nc = ncClass.nc_create(f, 'TKE', ('time', 'turbines'))
TKE_nc[:] = TKE

# Write to netcdf ref file
f4 = Dataset(outputFile_ref , 'w', format="NETCDF4")

f4.description = "WRF 3.8.1 - NEWA setup simulation: MYNN2.5 PBL: 27-9-3km 61 sigma levels. Time Series in wind turbines positions."

f4.createDimension('time', None)

# times
secs = ncClass.nc_create(f4, 'Times', ('time',))
secs.start_time = f1.variables['Times'].start_time
secs.end_time = f1.variables['Times'].end_time
secs[:] = timesSim

# Wind Components at 10m
U10_nc = ncClass.nc_create(f4, 'U10', ('time',))
U10_nc[:] = U10.mean(axis = 1)

V10_nc = ncClass.nc_create(f4, 'V10', ('time',))
V10_nc[:] = V10.mean(axis = 1)

# Wind Components at hub height
U_nc = ncClass.nc_create(f4, 'U', ('time',))
U_nc[:] = U.mean(axis = 1)

V_nc = ncClass.nc_create(f4, 'V', ('time',))
V_nc[:] = V.mean(axis = 1)

# Wind Components at hub height
U_corr_nc = ncClass.nc_create(f4, 'U_corr', ('time',))
U_corr_nc[:] = tsU_corr_df.values.mean(axis = 1)

V_corr_nc = ncClass.nc_create(f4, 'V_corr', ('time',))
V_corr_nc[:] = tsV_corr_df.values.mean(axis = 1)

# Wind Components at hub height
U_scada_nc = ncClass.nc_create(f4, 'U_scada', ('time',))
U_scada_nc[:] = tsU_scada_df.values.mean(axis = 1)

V_scada_nc = ncClass.nc_create(f4, 'V_scada', ('time',))
V_scada_nc[:] = tsV_scada_df.values.mean(axis = 1)

T2_nc = ncClass.nc_create(f4, 'T2', ('time',))
T2_nc[:] = T2.mean(axis = 1)

TSK_nc = ncClass.nc_create(f4, 'TSK', ('time',))
TSK_nc[:] = TSK.mean(axis = 1)

L_nc = ncClass.nc_create(f4, 'L', ('time',))
L_nc[:] = L.mean(axis = 1)

UST_nc = ncClass.nc_create(f4, 'UST', ('time',))
UST_nc[:] = UST.mean(axis = 1)

HFX_nc = ncClass.nc_create(f4, 'HFX', ('time',))
HFX_nc[:] = HFX.mean(axis = 1)

POT_nc = ncClass.nc_create(f4, 'POT', ('time',))
POT_nc[:] = POT.mean(axis = 1)

TKE_nc = ncClass.nc_create(f4, 'TKE', ('time',))
TKE_nc[:] = TKE.mean(axis = 1)

f.close()
f4.close()

f1.close()
f2.close()
f3.close()







