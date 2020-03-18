import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime
import pandas as pd
import netCDF4 
from scipy.interpolate import interp1d
from windrose import WindroseAxes
import matplotlib.cm as cm
from scipy.interpolate import interp1d
from scipy import stats
import seaborn as sns
from IPython.display import Markdown
import xarray as xr
from matplotlib import ticker as mtick
import matplotlib.gridspec as gridspec
import warnings
from scipy import interpolate
import alphashape
from shapely.geometry import Point, LineString
from tqdm import tqdm
import math
from scipy.linalg import lstsq

def Afreestream(P_gross,P,sims,bins,bins_label,turbines):
    """
    Estimate bias in freestream gross power for in each simulation by computing the ratio of the freestream power (Pfree), 
    as obtained from define_freestream function, and the gross power obtained from the mesoscale input (P_gross)
    Inputs:
        - P_gross: list of xarrays (wt, wd, zL) with gross power for each type of mesoscale input 
        - P: xarray of bin-averaged power at each turbine for all simulations (sim, wt, wd, zL) 
        - sims: dataframe with simulations description
        - bins: list of arrays defining the limits of each bin (wd, zL)
        - bins_label: list of arrays definind the labels of each bin (wt, wd, zL)
        - turbines: dataframe with turbines characteristics
    Outputs: 
        - Pfree: xarray of bin-averaged freestream power obtained from define_freestream function (wt, wd, zL)
        - Afree: xarray of bin-averaged freestream power to gross power ratio (wt, wd, zL)
    """
    WDbins, zLbins = bins
    Nwt, Nwd, NzL = [len(dim) for dim in bins_label] 
    wt_label, WDbins_label, zLbins_label = bins_label
    Nsim = sims.shape[0] 
    sims_label = sims['ID'].values

    # initialize outputs
    Pfree = xarray_init(['sim','wt','wd','zL'],[sims_label] + bins_label)
    Afree = xarray_init(['sim','wt','wd','zL'],[sims_label] + bins_label)

    points = turbines[['X coordinate','Y coordinate']].values
    pbar = tqdm(total = Nsim, leave = False, position = 0, desc="Progress") # progress bar
    for i_sim in range(Nsim):
        for i_wd in range(Nwd):
            for i_zL in range(NzL):
                WD = 0.5*(WDbins[i_wd] + WDbins[i_wd+1])
                freestream = define_freestream(points,WD)
                f1 = interpolate.NearestNDInterpolator(points[freestream], P[i_sim][freestream,i_wd,i_zL].values)
                Pfree[i_sim][:,i_wd,i_zL] = f1(points)
        if ("wakes-corr" in sims['Input'][i_sim]):
            Afree[i_sim] = Pfree[i_sim]/P_gross["wakes-corr"]
        if ("ctrl-corr" in sims['Input'][i_sim]):
            Afree[i_sim] = Pfree[i_sim]/P_gross["ctrl-corr"]
        if ("wakes" in sims['Input'][i_sim]):
            Afree[i_sim] = Pfree[i_sim]/P_gross["wakes"]
        if ("ctrl" in sims['Input'][i_sim]):
            Afree[i_sim] = Pfree[i_sim]/P_gross["ctrl"]
        if ("scada" in sims['Input'][i_sim]):
            Afree[i_sim] = Pfree[i_sim]/P_gross["scada"]
        pbar.update(1)
    pbar.close()
        
    return Pfree, Afree

def Ameso(file_obs,N_WDzL,binmap,bins_label,P_meso,file_Am,savefile = False, plot = True, figsize = (10,6)):
    """
    Compute/load and plot bin-averaged mesoscale correction factors 
    Inputs: 
        - file_obs: path to observed freestream power csv file
        - N_WDzL: dataframe wind bin sample count per wd and zL 
        - binmap: list of timestamp indices to samples in each bin (wd,zL)
        - P_meso: xarray of bin-averaged background gross power from mesoscale (wt, wd, zL)
        - file_Am: name of Am file to load/save
        - savefile: flag to save Am file 
        - plot: flag to return Am plot
    Outputs:
        - Am: xarray of bin-averaged mesoscale correction factors for each turbine (wt, wd, zL) 
        - Pfree_obs: xarray of bin-averaged observed "freestream" gross power from scada (wt, wd, zL)
        - Pfree_obs_std: xarray of bin standard deviation observed "freestream" gross power from scada (wt, wd, zL)
    """
    
    try:
        Nwt, Nwd, NzL = [len(dim) for dim in bins_label] 
        wt_label, WDbins_label, zLbins_label = bins_label

        Pfree_obs = xarray_init(['wt','wd','zL'],bins_label)
        Pfree_obs_std = xarray_init(['wt','wd','zL'],bins_label)

        ts = pd.read_csv(file_obs, index_col = 'time')
        times = netCDF4.num2date(ts.index,'seconds since 1970-01-01 00:00:00.00 UTC, calendar=gregorian')
        ts['time'] = times
        ts.set_index('time',inplace=True)
        ts = ts[wt_label] 
        Pfree_obs, Pfree_obs_std = bin_avrg(ts,binmap,bins_label)
        Am = Pfree_obs/P_meso   
        if savefile == True:
            Am.to_netcdf('./inputs/' + file_Am)
    except IOError:
        print ("No freestream data is available. Am loaded from ./inputs/%s " % (file_Am))
        Am = xr.open_dataset('./inputs/' + file_Am)
        Pfree_obs, Pfree_obs_std = None
    
    # Compute the total bias correction as a weighted averaged sum of all bins  
    Am_global = Am.mean(axis=0)
    Am_global_df = Am_global.to_dataframe(name = 'Am').unstack()
    Am_global_df.columns = Am_global_df.columns.get_level_values(1)
    Am_global_df = Am_global_df[['u','n','s']]
    Am_total = (Am_global_df*N_WDzL/N_WDzL.sum().sum()).sum().sum()
    print ("Total mesoscale bias correction = %.3f" % (Am_total))
    
    if plot:
        with sns.axes_style("darkgrid"):        
            Am_pd = pd.concat([Am[i,:,:].to_pandas() for i in range(Am.shape[2])])
            Am_pd.reset_index(level=0, inplace=True)
            Am_pd = pd.melt(Am_pd, id_vars=['wd'], value_vars=zLbins_label)
            Am_pd.rename(columns={'value':'Am'}, inplace=True)
            fig, ax = plt.subplots(figsize=figsize)
            ax = sns.barplot(x='wd', y='Am', hue='zL', data=Am_pd, palette = 'bwr_r',
                             edgecolor='grey',saturation=1) 
            ax.grid(True)
            ax.set_ylabel('$A_M$')
            ax.set_title('Wind farm mesoscale bias correction factor $A_M$')
            ax.legend(loc = 'upper left', bbox_to_anchor=(1, 1));
            sns.axes_style("white")
            
    return Am, Pfree_obs, Pfree_obs_std

def annotate_bars(ax,data,col,color = 'k'):
    """
    Add annotations in horizontal bar plots
    Inputs:
        - ax: axis handle
        - data: dataframe with data in bar plot
        - col: column of dataframe to annotate
        - color: text color
    """
    cnt = 0
    for index, row in data.iterrows():
        if row[col]<0: 
            ha = 'left'
        else:
            ha = 'right'
        ax.text(row[col],cnt+0.25, round(row[col],2), color=color, ha=ha)
        cnt = cnt + 1

def bin_avrg(ts,binmap,bins_label):
    """
    Compute bin statistics
    Inputs: 
        - ts: time-series
        - binmap: list of timestamp indices to samples in each bin (wd,zL)
        - bins_label: list of lists of labels for each bin 
    Outputs: 
        - mean: bin-average
        - std:  bin standard deviation
    """
    if len(bins_label) == 2: 
        Nwd, NzL = [len(dim) for dim in bins_label]
        WDbins_label, zLbins_label = bins_label
        mean = xarray_init(['wd','zL'],bins_label)
        std = xarray_init(['wd','zL'],bins_label)        
        for i_wd in range(Nwd):
            for i_zL in range(NzL):
                ts_bin = ts.reindex(binmap[i_wd,i_zL])
                mean[i_wd,i_zL] = ts_bin.mean()
                std[i_wd,i_zL] = ts_bin.std()
    else:
        Nwt, Nwd, NzL = [len(dim) for dim in bins_label]
        wt_label, WDbins_label, zLbins_label = bins_label
        mean = xarray_init(['wt','wd','zL'],bins_label)
        std = xarray_init(['wt','wd','zL'],bins_label)
        for i_wd in range(Nwd):
            for i_zL in range(NzL):
                ts_bin = ts.reindex(binmap[i_wd,i_zL])
                mean[:,i_wd,i_zL] = ts_bin.mean()
                std[:,i_wd,i_zL] = ts_bin.std()
    return mean, std

def centroid(arr):
    """
    Define coordinates of the centroid based on list of turbine coordinates
    Inputs:
        - arr: array with list of coordinates [x,y]
    Outputs: 
        - x,y: centroid coordinates 
    """
    
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    
    return sum_x/length, sum_y/length # average 

def define_benchmark(validation, validation_data, benchmark,P_obs,P_obs_std,P,P_std, sims):
    """
    Define benchmark data to compute metrics
    Inputs: 
        - validation:  True if the benchmark is based on observations 
        - validation_data: True if observations data exist (output from read_obs)
        - benchmark: ID of simulation to be used as benchmark for code-to-code assessment
        - P_obs: xarray of bin-averaged observed power at each turbine (wt, wd, zL)
        - P_obs_std: xarray of bin standard deviation of observed power at each turbine (wt, wd, zL)
        - P: xarray of bin-averaged power at each turbine for all simulations (sim, wt, wd, zL) 
        - Pstd: xarrary of bin standard deviation of power at each turbine for all simulations (sim, wt, wd, zL)
        - sims: dataframe with simulations description
    Outpus:
        - P_bench: xarray of bin-averaged benchmark power at each turbine (wt, wd, zL)
        - P_bench_std: xarray of bin standard deviation benchmark power at each turbine (wt, wd, zL)
    """
    
    if (validation and validation_data):
        print ("You are doing validation against observations")
        P_bench = P_obs
        P_bench_std = P_obs_std
    else: 
        print ("You are doing code-to-code assessment against the benchmark series $%s$ " % (sims["Label"][benchmark]))
        P_bench = P[benchmark]
        P_bench_std = P_std[benchmark]
    return P_bench, P_bench_std

def define_freestream(points,WD):
    """
    Define which turbines in the layout are in freestream conditions
    Inputs:
        - points: array with x,y coordinates [m]
        - WD: wind direction 
    Outputs:
        - freestream: boolean array with freestream turbines = True   
    """
    # raylength is the distance [m] upstream from the centroid to do ray-casting along a line 
    # that passes through each turbine in the boundary of the alphashape. 
    # Make sure the ray is always out of the wind farm
    raylength = 30000. 
    sectorwidth = 45. # wind direction sector [deg] free of upstream turbines around the input wind direction 
    alpha = 0.0004    # controls the tightness of the bounding shape to the layout (alpha = 0 returns the convex hull)
    Nwt = points.shape[0] # number of turbines
    
    alpha_shape = alphashape.alphashape(points, alpha) # polygon of the wind farm to define turbines in freestream conditions (= turbines in the boundary)
    freestream = np.zeros(Nwt, dtype=bool)
    wd = np.array([WD - sectorwidth/2, WD + sectorwidth/2])
    polarangle = (-wd + 90.)*np.pi/180. # from wind direction to polar angle
    
    for wt in range(Nwt):
        point_turbine = Point(points[wt,:])
        point_upstream1 = Point(point_turbine.x + raylength*np.cos(polarangle[0]), 
                                point_turbine.y + raylength*np.sin(polarangle[0]))
        point_upstream2 = Point(point_turbine.x + raylength*np.cos(polarangle[1]), 
                                point_turbine.y + raylength*np.sin(polarangle[1]))
        if alpha_shape.exterior.distance(point_turbine) <= 10.: # turbine in the boundary
            line1 = LineString([point_upstream1, point_turbine]) 
            line2 = LineString([point_upstream2, point_turbine])
            if (not line1.crosses(alpha_shape.exterior)) and (not line2.crosses(alpha_shape.exterior)):
                freestream[wt] = True
                
    return freestream

def flags_to_ts(scada_flags, min_data_availability):
    """
    Function to filter timestamps with high-quality data for validation. 
    Inputs: 
        - scada_flags: timeseries of flags, where 0 means original good data and 1 means wrong/missing data. 
                       Fractions between 0 and 1 indicate incomplete data, for example two missing 10-min samples 
                       in an hourly interval would be indicated with a 2/6 value 
        - min_data_availability: minimum availability of original observational data
    Outputs:
        - goodtimestamps: list the time stamps that will be retained in the analysis
    """
    
    filter_condition = (100 - min_data_availability)/100 * scada_flags.shape[1]
    flags_filter = (scada_flags.sum(axis=1) < filter_condition) 
    goodtimestamps = netCDF4.num2date(flags_filter[flags_filter].keys(),
                                      'seconds since 1970-01-01 00:00:00.00 UTC, calendar=gregorian')
    
    return goodtimestamps

def S_grad(x,y,U,V,coord = 'uv'):
    """
    Compute non-dimensional horizontal gradients accross the wind farm by fitting the horizontal wind speed to a plane
    Inputs:
        - x,y: turbine coordinates in meters
        - U,V: velocity components in natural coordinates
        - coord: coordinate system
                    'uv': wind direction coordinates
                    'xy': natural coordinates
    Outputs:
        - Sgrad: Dataframe with gradient components and magnitude. The gradients are returned in non-dimensional form
                by multiplying by the length of the wind farm and dividing by the mean wind speed
    """
    # Least-squares fit of the horizontal wind speed to a plane: S = p[0]*x + p[1]*y + p[2]
    U = U.dropna()
    V = V.dropna()
    S = (U**2 + V**2)**0.5 
    WD = (270-np.rad2deg(np.arctan2(V,U)))%360
    n = len(x)
    b = S.T
    A = np.array([x,y,np.ones(n)]).T
    p, res, rnk, s = lstsq(A, b) 
    
    # test 
    #from mpl_toolkits.mplot3d import Axes3D
    #xx, yy = np.meshgrid(np.linspace(x_wt.min(), x_wt.max(), 101), 
    #                    np.linspace(y_wt.min(), y_wt.max(), 101))
    #t = 100 # choose a timestamp index
    #zz = p[0,t]*xx + p[1,t]*yy + p[2,t]
    #fig = plt.figure()
    #ax = fig.gca(projection='3d')
    #ax.plot_surface(xx, yy, zz, alpha=0.2,label='least squares fit, $S = p0 + p1*x + p2*y$')
    #ax.scatter(x, y, b[:,t], marker='o', label='data')
    #ax.set_xlabel('x')
    #ax.set_ylabel('y')
    #ax.set_zlabel('S')
    #plt.grid(alpha=0.25)
    #plt.show()
    
    if coord == 'xy': # natural coordinates
        Sgrad = pd.DataFrame(index=S.index, columns = {'Sx','Sy','Sh'})
        Sgrad['Sx'] = p[0,:] # East component
        Sgrad['Sy'] = p[1,:] # North component
        Sgrad['Sh'] = (p[0,:]**2 + p[1,:]**2)**0.5 # magnitude
    else: # wind direction coordinates
        theta = np.arctan(V/U).mean(axis = 1)  
        Sgrad = pd.DataFrame(index=S.index, columns = {'Su','Sv','Sh'})
        Sgrad['Su'] = (p[0,:]*np.cos(theta) + p[1,:]*np.sin(theta)) # streamwise component 
        Sgrad['Sv'] = -p[0,:]*np.sin(theta) + p[1,:]*np.cos(theta) # cross-wind component 
        Sgrad['Sh'] = (p[0,:]**2 + p[1,:]**2)**0.5 # magnitude
        
    points = pd.concat([x,y], axis = 1).values
    length = windfarmlength(points, WD.median(axis = 1))
        
    Sgrad = Sgrad.multiply(length/S.mean(axis = 1).values, axis=0) # non-dimensional gradient 
    
    return Sgrad

def longformat(data,sims,tags,id_vars,value_vars,metric_label):
    """
    Trasnsform xarray to long-format dataframes with simulation tags for seaborn plots y = f(x)
    Inputs:
        - data: xarray of dimensions [sim, wd, zL]
        - sims: dataframe with simulations description
        - tags: simulation identifier
        - id_vars: independent variabels ("x")
        - value_vars: dependent variable ("y")
        - metric_label: label of the dependent variable
    Outputs: 
        - df: long-format dataframe with dependent and independent variables for seaborn plots
    """
    
    # filter simulations that will be displayed in the plot
    plotresults = sims.index[sims['Plot'] == 1].tolist() 
    df = data[plotresults].to_pandas()  
    
    # add information about the simulation identifier 
    df['sim'] = sims.loc[plotresults][tags].values 
    df['Input'] = sims.loc[plotresults]['Input'].values
    df['Model'] = sims.loc[plotresults]['Model Type'].values
    #for i_sim in range(len(plotresults)): # add wake decay coefficient by searching "K = " in the Remarks column
    #    data.loc[sims['ID'][plotresults[i_sim]],'K'] = sims.loc[plotresults[i_sim]]['Remarks'].partition('K = ')[2]    
    
    df = pd.melt(df, id_vars=id_vars, value_vars=value_vars) # unpivot a DataFrame from wide to long format
    df.rename(columns={'value':metric_label}, inplace=True)  
    
    return df

def overall_metrics(eta_farm,bias_farm,mae_farm,sims,N_WDzL,Nmin,binmap,tags,figsize = (10,6), plot = True):   
    """
    Compute overall metrics using weighted bin-averages for all simulations
    Inputs: 
        - eta_farm:  xarray of simulated bin-averaged wind farm array efficiency (sim, wd, zL)
        - bias_farm: xarray of wind farm array efficiency bias for each simulation (sim, wd, zL)
        - mae_farm: xarray of wind farm array efficiency mean absolute error for each simulation (sim, wd, zL)
        - sims: dataframe with simulations description
        - N_WDzL: dataframe wind bin sample count per wd and zL (wd, zL)
        - Nmin: minimum number of samples to include the bin in the integration
        - binmap: list of timestamp indices to samples in each bin (wd,zL)
        - tags: list of simulation names 
        - figsize: figure size
        - plot: flag to return a figure
    Outputs:
        - metrics: dataframe with overall wind farm metrics 
    """
    
    n_sim = len(sims)
    plotresults = sims.index[sims['Plot'] == 1].tolist()
    len_array = np.vectorize(len)
    bin_sizes = len_array(binmap)
    bin_sizes[N_WDzL < Nmin] = 0      # filter out bins with less than Nmin samples (weight = 0)
    sum_bin_sizes = bin_sizes.sum()

    eta_farm_tot = np.sum(eta_farm*bin_sizes/sum_bin_sizes, axis = (1,2)) 
    bias_farm_tot = np.sum(bias_farm*bin_sizes/sum_bin_sizes, axis = (1,2)) 
    mae_farm_tot = np.sum(mae_farm*bin_sizes/sum_bin_sizes, axis = (1,2)) 

    # if there are no result for all bins don't compute the total values
    notallbins = np.any(np.isnan(eta_farm), axis = (1,2))
    bias_farm_tot[notallbins] = np.nan
    mae_farm_tot[notallbins] = np.nan
    eta_farm_tot[notallbins] = np.nan
    
    # initialize output dataframe
    metrics = pd.DataFrame(np.zeros((n_sim,6)),
                           columns = ['sim','Input','ArrayEff[%]','WakeLoss[%]','BIAS[%]','MAE[%]'],
                           index = sims.index) #sims[tags]
    for i_sim in range(n_sim):
        metrics.loc[i_sim,'sim'] = sims[tags][i_sim] 
        metrics.loc[i_sim,'Input'] = sims['Input'][i_sim] 
        metrics.loc[i_sim,'ArrayEff[%]'] = eta_farm_tot[i_sim] #sims[tags][isim]
        metrics.loc[i_sim,'WakeLoss[%]'] = 100. - eta_farm_tot[i_sim] #sims[tags][isim]
        metrics.loc[i_sim,'BIAS[%]'] = bias_farm_tot[i_sim]
        metrics.loc[i_sim,'MAE[%]'] = mae_farm_tot[i_sim]

    if plot:
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(nrows=1, ncols=3, sharey='all',figsize=figsize)
            data = metrics.iloc[plotresults].dropna()
            ax[0] = sns.barplot(x="ArrayEff[%]", y="sim", hue="Input", data=data,
                        palette="muted", dodge=False, ax = ax[0])
            ax[0].get_legend().remove()
            ax[0].set_ylabel('')
            ax[0].grid(True)
            annotate_bars(ax[0],data,"ArrayEff[%]",color = 'k')

            ax[1] = sns.barplot(x="BIAS[%]", y="sim", hue="Input", data=data,
                        palette="muted", dodge=False, ax = ax[1])
            ax[1].get_legend().remove()
            ax[1].grid(True)
            ax[1].set_ylabel('')    
            annotate_bars(ax[1],data,"BIAS[%]", color = 'k')

            ax[2] = sns.barplot(x="MAE[%]", y="sim", hue="Input", data=data,
                        palette="muted", dodge=False, ax = ax[2])
            ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[2].grid(True)
            ax[2].set_ylabel('')
            annotate_bars(ax[2],data,"MAE[%]", color = 'k')
                
                
        sns.axes_style("white")
    return metrics

def plot_Afree(Afree,sim,N_WDzL,zLbins_label):
    """
    Plot bin-averaged Afree, the ratio of the freestream power to the gross power, for a simulation
    Inputs: 
        - Afree: xarray of bin-averaged freestream power to gross power ratio (wt, wd, zL)
        - sim: ID of simulation to be ploted
        - N_WDzL: dataframe wind bin sample count per wd and zL (wd, zL)
        - zLbins_label: array with labels of stability bins (zL)
    Outputs: 
        - ax: axis handle
    """
    with sns.axes_style("darkgrid"):        
        Afree_pd = pd.concat([Afree.loc[sim][i,:,:].to_pandas() for i in range(Afree.loc[sim].shape[2])])
        Afree_pd.reset_index(level=0, inplace=True)
        Afree_pd = pd.melt(Afree_pd, id_vars=['wd'], value_vars=zLbins_label)
        Afree_pd.rename(columns={'value':'Afree'}, inplace=True)
        fig, ax = plt.subplots(figsize=(10,6))
        ax = sns.barplot(x='wd', y='Afree', hue='zL', data=Afree_pd, palette = 'bwr_r',
                         edgecolor='grey',saturation=1) #sns.color_palette("coolwarm", 3)
        ax.grid(True)
        ax.set_title('Wind farm freestream bias correction factor $A_{free}$: ' + sim)
        ax.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
        ax.set_ylim((0,1.2))
        sns.axes_style("white")
    
    # Compute the total freestream ratio as a weighted averaged sum of all bins 
    Afree_global = Afree.loc[sim].mean(axis=0).drop('sim')
    Afree_global_df = Afree_global.to_dataframe(name = 'Afree').unstack()
    Afree_global_df.columns = Afree_global_df.columns.get_level_values(1)
    Afree_global_df = Afree_global_df[zLbins_label]

    Afree_total = (Afree_global_df*N_WDzL/N_WDzL.sum().sum()).sum().sum()
    print("Total freestream bias correction = %.3f" % (Afree_total))
    
    return ax
   
def plot_eta_WD_zL(sim_eta, bias, bench_eta, sims, N_WDzL, validation, plot_type = 'highlight'):  
    """
    Plot array efficiency and bias per wind direction and stability 
    Inputs:
        - sim_eta: simulated wind farm array efficiency. xarray [sim, wd, zL]
        - bias: wind farm array efficiency bias. xarray [sim, wd, zL]
        - bench_eta: benchmark array efficiency used to compute the bias. xarray [wd, zL]
        - sims: dataframe with simulations description
        - N_WDzL: dataframe wind bin sample count per wd and zL 
        - validation: validation flag (True if observational data is used in bench_eta) 
        - plot_type: type of plot, 
                    'highlight' will plot a subset of simulations  
                    'cat' will combine the simulations based on input data 
    -Outputs:  
        - f1_ax1, f1_ax2, f1_ax3, f1_ax4: axes handles
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore RuntimeWarning warning 
        
        # filter simulations that will be displayed in the plot
        plotresults = sims.index[sims['Plot'] == 1].tolist() 
        highlight = sims.index[sims['Highlight'] == 1].tolist()
        tags = 'Label' # which column in sims will be used to tag the simulations

        eta_zL = sim_eta.mean(axis = 1, skipna = True) # array efficiency vs wind direction 
        bias_zL = bias.mean(axis = 1, skipna = True)   # bias vs wind direction [%] 
        N_zL = N_WDzL.sum(axis = 0, skipna = True) # sample count
        N_zL_max = int(math.ceil(max(N_zL) / 200.0)) * 200

        eta_WD = sim_eta.mean(axis = 2, skipna = True) # array efficiency vs wind direction 
        bias_WD = bias.mean(axis = 2, skipna = True)   # bias vs wind direction [%] 
        N_WD = N_WDzL.sum(axis = 1, skipna = True) # sample count
        N_WD_max = int(math.ceil(max(N_WD) / 200.0)) * 200

        zLbins_label = sim_eta.coords['zL'].values.tolist()
        WDbins_label = sim_eta.coords['wd'].values.tolist()
        
        # from wide to long-formatted dataframes
        eta_zL = longformat(eta_zL,sims,tags,['sim','Input','Model'],zLbins_label,"ArrayEff[%]")
        bias_zL = longformat(bias_zL,sims,tags,['sim','Input','Model'],zLbins_label,"BIAS[%]")
        eta_WD = longformat(eta_WD,sims,tags,['sim','Input','Model'],WDbins_label,"ArrayEff[%]")
        bias_WD = longformat(bias_WD,sims,tags,['sim','Input','Model'],WDbins_label,"BIAS[%]")

        # linestyles for spagetty plots
        linestyles = ['-', '--', ':', '-.','o', '^', 'v', '<', '>', 's',
              '+', 'x', 'd', '1', '2', '3', '4', 'h', 'p', '|', '_', 'D', 'H']
        
        f1 = plt.figure(figsize = (14,8), constrained_layout = True)   
        spec = gridspec.GridSpec(ncols=2, nrows=2, figure=f1, width_ratios=[2, 0.6], height_ratios=[2, 1])    
        
        f1_ax1 = f1.add_subplot(spec[0, 0])
        if (highlight) and ('highlight' in plot_type):
            eta_highlight_WD = sim_eta[highlight].mean(axis = 2, skipna = True).to_pandas()
            eta_highlight_WD[tags] = sims.iloc[highlight][tags].values
            eta_highlight_WD = eta_highlight_WD.set_index(tags)
            eta_highlight_WD.T.plot(grid=1, linewidth = 2, colormap = 'Dark2', 
                                    style = linestyles, legend=False, ax = f1_ax1)
        if validation:
            eta_obs_WD = bench_eta.mean(axis = 1, skipna = True).to_pandas()
            eta_obs_WD.plot(grid=1, marker='s', markerfacecolor='grey', linewidth = 3, markeredgecolor= 'k', 
                            markersize = 8, color = 'k', linestyle='--', label = 'scada', ax = f1_ax1)
        if ('cat' in plot_type):
            f1_ax1 = sns.lineplot(x='wd', y='ArrayEff[%]', hue='Input', data=eta_WD, 
                              sort=False, legend=False, ax = f1_ax1) # , style = 'Model' (only 6 style categories allowed)
        f1_ax1.set_xticklabels(WDbins_label)
        f1_ax1.set_ylabel('Array Efficiency [%]')
        f1_ax1.grid(True)
        #f1_ax1.get_legend().remove()
        f1_ax1.set_ylim([40,100])
        f1_ax1b=f1_ax1.twinx()
        N_WD.plot.bar(color = 'silver', ax = f1_ax1b)
        f1_ax1b.set_yticks(np.linspace(0, N_WD_max, 5).tolist())
        f1_ax1b.set_ylim([0,N_WD_max*3])

        f1_ax2 = f1.add_subplot(spec[1, 0], sharex=f1_ax1)
        f1_ax2 = sns.barplot(x='wd', y='BIAS[%]', hue='Input', data=bias_WD, ax = f1_ax2)
        f1_ax2.set_ylabel('BIAS [%]')
        f1_ax2.get_legend().remove()
        f1_ax2.set_ylim([-15,15])
        f1_ax2.set_xlabel('')
        f1_ax2.grid(True)

        f1_ax3 = f1.add_subplot(spec[0, 1], sharey=f1_ax1)
        if (highlight) and ('highlight' in plot_type):
            eta_highlight_zL = sim_eta[highlight].mean(axis = 1, skipna = True).to_pandas()
            eta_highlight_zL[tags] = sims.iloc[highlight][tags].values
            eta_highlight_zL = eta_highlight_zL.set_index(tags)
            eta_highlight_zL.T.plot(grid=1, linewidth = 2, colormap = 'Dark2', 
                                    style = linestyles, legend=False, ax = f1_ax3)
        if validation:
            eta_obs_zL = bench_eta.mean(axis = 0, skipna = True).to_pandas()
            eta_obs_zL.plot(grid=1, marker='s', markerfacecolor='grey', linewidth = 2, markeredgecolor= 'k', 
                            markersize = 8, color = 'k', linestyle='--', label = 'scada', ax = f1_ax3)
        if ('cat' in plot_type):
            f1_ax3 = sns.lineplot(x='zL', y='ArrayEff[%]', hue='Input', data=eta_zL, 
                              sort=False, ax = f1_ax3) # , style = 'Model' (only 6 style categories allowed)
        f1_ax3.grid(True)
        f1_ax3.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
        f1_ax3b=f1_ax3.twinx()
        N_zL.plot.bar(color = 'silver', ax = f1_ax3b)
        f1_ax3b.set_yticks(np.linspace(0, N_zL_max, 5).tolist())
        f1_ax3b.set_ylim([0,N_zL_max*3])
        f1_ax3b.set_ylabel('Samples')
        f1_ax3b.yaxis.set_label_coords(1.2,0.1)
        f1_ax3.set_xticklabels('')
        f1_ax3.set_xlabel('')

        f1_ax4 = f1.add_subplot(spec[1, 1], sharex=f1_ax3, sharey=f1_ax2)
        f1_ax4 = sns.barplot(x='zL', y='BIAS[%]', hue='Input', data=bias_zL, ax = f1_ax4)
        f1_ax4.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
        #f1_ax4.set_yticklabels('')
        f1_ax4.set_ylabel('')
        f1_ax4.set_xlabel('')
        f1_ax4.grid(True)

    return f1_ax1, f1_ax2, f1_ax3, f1_ax4

def plot_gradient(S_grad, var = 'h', figsize = (10,6), ylim = [0.6,1.4]):
    """ 
    Plot gradients per wind direction and stability
    Inputs: 
        - S_grad: dataframe with horizontal gradient
        - figsize:  figure size
        - var: variable to plot {'u','v','h'}
        - ylim: y-axis limits
    """
    with sns.axes_style("darkgrid"): 
        if var == 'P':
            value = 'Pdelta'
            title = 'Gross Power Variability'
            ylabel = '$\Delta P_{gross}$ [%]'
        elif var == 'u':
            value = 'S_u'
            title = 'Longitudinal Wind Speed Gradient'
            ylabel = '$S_{u}L_{u}/S$'
        elif var == 'v':
            value = 'S_v'
            title = 'Transversal Wind Speed Gradient'
            ylabel = '$S_{v}L_{u}/S$'
        elif var == 'h':
            value = 'S_grad'
            title = 'Horizontal Wind Speed Gradient'
            ylabel = '$|grad(S)|L_{u}/S$'        
        S_grad_pd = S_grad.to_pandas()
        zLbins_label = S_grad_pd.columns.values
        S_grad_pd.reset_index(level=0, inplace=True)
        S_grad_pd = pd.melt(S_grad_pd, id_vars=['wd'], value_vars=zLbins_label)
        S_grad_pd.rename(columns={'value':value}, inplace=True)
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.barplot(x='wd', y=value, hue='zL', data=S_grad_pd, palette = 'bwr_r',
                         edgecolor='grey',saturation=1) 
        ax.grid(True)
        ax.set_title(title)
        ax.legend(loc = 'upper left', bbox_to_anchor=(1, 1));
        ax.set_ylabel(ylabel)
        ax.set_ylim(ylim)

def plot_heatmaps(data,sims, N_WDzL, Nmin, ax_size = (1.7,4), cols = 4):
    """
    Plot heatmaps of bin metrics for all simulation in plotresults
    Inputs:
        - data: xarray of bin-averaged metric for all simulations (sim, wd,zL)
        - sims: dataframe with simulations description
        - N_WDzL: dataframe wind bin sample count per wd and zL (wd, zL)
        - Nmin: minimum number of samples to include the bin in the integration
        - ax_size: size of axis
        - cols: number of columns in grid plot    
    """
    
    plotresults = sims.index[sims['Plot'] == 1].tolist()
    data = data[plotresults]
    mask = np.zeros_like(N_WDzL)
    mask[N_WDzL < Nmin] = True # mask will plot bins with less than Nmin samples in grey color

    sim_id = data.coords['sim'].values.flatten()

    n_sim = len(data)

    if n_sim < cols:
        cols = n_sim

    rows = int(n_sim / cols)
    if rows < (n_sim / cols): 
        rows = rows+1

    with sns.axes_style("darkgrid"):
        fig, ax = plt.subplots(rows, cols, sharex='col', sharey='row', 
                               figsize=(cols*ax_size[0] ,rows*ax_size[1]))
        for iax in range (0,n_sim):
            if rows>1:
                index = np.unravel_index(iax,(rows,cols))
            else: 
                index = iax
            #colour bar for the last plot in row
            #if iax%cols == (cols-1):
            #    cbar = True
            #else:
            #    cbar = False
            sns.heatmap(data[iax].to_pandas(), ax = ax[index], annot=True, center=0, 
                        linewidths=.1, xticklabels = True, yticklabels=True,
                        cmap="bwr_r",cbar = False, mask = mask) # use annotation and not colorbar
            ax[index].set_title(sim_id[iax])
            ax[index].set_xlabel('')
            ax[index].set_ylabel('')
        plt.tight_layout()
    sns.axes_style("white");    
        
def plot_layout_timestamp():
    """
    Layout plot to visualize different quantities at a specific timestamp
    """
    time0 = scada_ts[10]  
    # 45 : NW farm-farm effects visible and not fully captured by obs_Pfree missing on the longitudinal gradients
    datetime0 = datetime.datetime.fromtimestamp(time0)
    Pscada = obs_ts.loc[time0]
    Pfree = obs_Pfree_ts.loc[time0]
    Pmeso = meso_ts_power.loc[time0]
    Psim = sim_ts[7].loc[time0]
    WDref = mast.data['WD'].loc[datetime0][0]
    Sref = mast.data['S'].loc[datetime0][0]

    vmin = 2
    vmax = 4
    x = turbines['X coordinate']-x_ref
    y = turbines['Y coordinate']-y_ref
    cmap=plt.cm.get_cmap('jet',8)
    fig, ax = plt.subplots(2,2,figsize=(12,12))

    sc = ax[0,0].scatter(x,y,marker='o',c=Pfree,cmap=cmap,edgecolors ='k', vmin = vmin, vmax=vmax)
    labels = list(map("{:.2f}".format,Pfree.values))
    for i in range(len(labels)):
        ax[0,0].text(x[i]+50, y[i]+50, labels[i], fontsize=9) 
    ax[0,0].set_aspect(1.0)
    ax[0,0].set_title('Pfree')

    sc = ax[0,1].scatter(x,y,marker='o',c=Pmeso,cmap=cmap,edgecolors ='k', vmin = vmin, vmax=vmax)
    labels = list(map("{:.2f}".format,Pmeso.values))
    for i in range(len(labels)):
        ax[0,1].text(x[i]+50, y[i]+50, labels[i], fontsize=9) 
    ax[0,1].set_aspect(1.0)
    ax[0,1].set_title('Pmeso')

    sc = ax[1,0].scatter(x,y,marker='o',c=Pscada,cmap=cmap,edgecolors ='k', vmin = vmin, vmax=vmax)
    labels = list(map("{:.2f}".format,Pscada.values))
    for i in range(len(labels)):
        ax[1,0].text(x[i]+50, y[i]+50, labels[i], fontsize=9) 
    ax[1,0].set_aspect(1.0)
    ax[1,0].set_title('Pscada')

    sc = ax[1,1].scatter(x,y,marker='o',c=Psim,cmap=cmap,edgecolors ='k', vmin = vmin, vmax=vmax)
    labels = list(map("{:.2f}".format,Psim.values))
    for i in range(len(labels)):
        ax[1,1].text(x[i]+50, y[i]+50, labels[i], fontsize=9) 
    ax[1,1].set_aspect(1.0)
    ax[1,1].set_title('Psim')
    #plt.colorbar(sc,ax=ax3)
    print('{}      WDref = {:.2f}, Sref = {:.2f}'.format(datetime0, WDref,Sref))
    
def plot_transect(data,bench_data,P_gross,WDbin,zLbin,wt_list,turbines,sims,plotresults,highlight,validation, ylim1 = [0.6,1.4], ylim2 = [0.9,1.1], figsize = (16,8), plot_type = 'highlight'):
    """
    Plot array efficiency and gross power data along a transect of turbines
    Inputs: 
        - data: simulated array efficiency data. xarray [sim, wt, wd, zL]
        - bench_data: benchmark array efficiency data to compare against. xarray [wt, wd, zL]
        - P_gross: list of turbine gross power for each type of input. list of xarrays [wt, wd, zL]
        - WDbin: wind direction bins
        - zLbin: stability bins
        - wt_list: list of wind turbines in the transect
        - turbines: dataframe with turbines characteristics
        - sims: dataframe with simulations description
        - plotresults: simulations to be plotted
        - highlight: simulations to be highlighted
        - validation: validation flag (True if observational data is used in bench_eta) 
        - ylim1: y limits for top axis
        - ylim2: y limits for bottom axis
        - figsize: figure size
        - plot_type: type of plot, 
                    'highlight' will plot a subset of simulations  
                    'cat' will combine the simulations based on input data 
    """
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Normalize based on wtnorm position
#        if wtnorm == 'ref':
#            datanorm = data.loc[:,wt_list]/data.mean(axis=1)
#            benchdatanorm = bench_data.loc[wt_list]/bench_data.mean()
#        else:
#            datanorm = data.loc[:,wt_list]/data.loc[:,wtnorm]
#            benchdatanorm = bench_data.loc[wt_list]/bench_data.loc[wtnorm]

        # Array efficiency data is already normalized
        datanorm = data.loc[:,wt_list,WDbin,zLbin]
        benchdatanorm = bench_data.loc[wt_list,WDbin,zLbin]

        # from wide to long-formatted dataframes
        tags = 'Label'
        datanorm_sns = longformat(datanorm,sims,tags,['sim','Input','Model'],wt_list,"eta")

        n_wt = len(wt_list)
        
        x_ref, y_ref = centroid(turbines[['X coordinate','Y coordinate']].values) # coordinates of wind farm centroid
        x_wt = turbines['X coordinate']-x_ref
        y_wt = turbines['Y coordinate']-y_ref

        # linestyles for spagetty plots
        linestyles = ['-', '--', ':', '-.','o', '^', 'v', '<', '>', 's',
              '+', 'x', 'd', '1', '2', '3', '4', 'h', 'p', '|', '_', 'D', 'H']

        f1 = plt.figure(figsize = figsize)   
        spec = gridspec.GridSpec(ncols=2, nrows=2, figure=f1, width_ratios=[1, 1.5], height_ratios=[2, 1])    
        
        # plot layout with transect turbines highlighted
        iwt = [x in wt_list for x in turbines['VDC ID']]
        f1_ax1 = f1.add_subplot(spec[:, 0])
        f1_ax1.scatter(x_wt,y_wt,c='silver', marker = 'o', s=20)
        f1_ax1.scatter(x_wt[iwt],y_wt[iwt],c='black', marker = 'o',s=20)
        f1_ax1.text(x_wt.loc[iwt].iloc[0],y_wt.loc[iwt].iloc[0],turbines['VDC ID'].iloc[iwt].iloc[0],
                    {'ha': 'right'}, fontsize=14)
        f1_ax1.text(x_wt.loc[iwt].iloc[-1],y_wt.loc[iwt].iloc[-1],turbines['VDC ID'].iloc[iwt].iloc[-1],
                    {'ha': 'right'}, fontsize=14)
        f1_ax1.axis('scaled')
        f1_ax1.spines['top'].set_visible(False)
        f1_ax1.spines['bottom'].set_visible(False)
        f1_ax1.spines['left'].set_visible(False)
        f1_ax1.spines['right'].set_visible(False)
        f1_ax1.get_xaxis().set_ticks([])
        f1_ax1.get_yaxis().set_ticks([])
       
        # plot array efficiency 
        f1_ax2 = f1.add_subplot(spec[0, 1])
        if (highlight) and ('highlight' in plot_type):
            datanorm_pd = datanorm[highlight].to_pandas()
            datanorm_pd[tags] = sims.iloc[highlight][tags].values
            datanorm_pd = datanorm_pd.set_index(tags)
            datanorm_pd.T.plot(grid=1, linewidth = 2, colormap = 'Dark2', 
                                    style = linestyles, legend=False, ax = f1_ax2)
        if validation: # plot observational data 
            f1_ax22 = f1_ax2.twinx()
            benchdatanorm.to_pandas().plot(marker='s', markerfacecolor='grey', linewidth = 2, markeredgecolor= 'k', 
                             markersize = 8, color = 'k', linestyle='--', ax = f1_ax22, label = 'scada')
            f1_ax22.set_ylim(ylim1)
            f1_ax22.set_yticklabels('')
            f1_ax22.set_yticks([]) 
        if ('cat' in plot_type):
            f1_ax2 = sns.lineplot(x='wt', y='eta', hue='Input', data=datanorm_sns, sort=False) #, style = 'Model'
        #f1_ax2.set_xticklabels(wt_list)
        f1_ax2.legend(loc = 'upper left', bbox_to_anchor=(1, 1)) 
        f1_ax2.set_ylim(ylim1)
        f1_ax2.set_xlabel(None)
        f1_ax2.set_ylabel('Array Efficiency')  
        f1_ax2.set_title('Transect '+wt_list[0]+'-'+wt_list[-1]+' ('+WDbin+', '+zLbin+')')
        f1_ax2.grid(True)
        f1_ax2.margins(0.05)
        xlim1 = f1_ax2.get_xlim()
        f1_ax2.set_xlim([xlim1[0]-0.5,xlim1[1]+0.5])

        # plot gross power ratio
        f1_ax3 = f1.add_subplot(spec[1, 1], sharex = f1_ax2)
        linestyle = {"ctrl": '-b',"ctrl-corr": '-.r',"wakes": '--g', "wakes-corr": '-.r', "scada": 's-k'}
        for p in P_gross:
            # normalize gross power by dividing with the mean of the wind turbines gross power (ref) 
            P_gross_ratio = P_gross[p].loc[wt_list,WDbin,zLbin]/P_gross[p].loc[:,WDbin,zLbin].mean()
            if validation and p == "scada":
                f1_ax3.plot(wt_list,P_gross_ratio,marker='s', markerfacecolor='grey', linewidth = 2,
                            markeredgecolor= 'k',  markersize = 8, color = 'k', linestyle='--', label = 'scada')
            else:
                f1_ax3.plot(wt_list,P_gross_ratio,linestyle[p],linewidth = 2,label = p)
        f1_ax3.set_ylabel('Gross Power Ratio') 

        f1_ax3.set_ylim(ylim2)
        f1_ax3.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
        f1_ax3.grid(True)
        

    return f1_ax1,f1_ax2,f1_ax3

def plot_wf_layout(x, y, data, labels = [], vmin = np.nan, vmax = np.nan, cmap = 'jet', title = [], figsize=(12,6),ax = None):  
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if len(data) == 0:
        sc = ax.scatter(x,y,marker='o',c='b',edgecolors ='k')
    else:
        cmap=plt.cm.get_cmap(cmap,9)
        sc = ax.scatter(x,y,marker='o',c=data,cmap=cmap,edgecolors ='k', vmin = vmin, vmax=vmax)
        plt.colorbar(sc,ax=ax)
    for i in range(len(labels)):
        ax.text(x[i]+50, y[i]+50, labels[i], fontsize=9)
    ax.set_aspect(1.0)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.margins(0.1)
    ax.set_title(title)
    ax.grid()
    return sc

def read_input(file,datefrom,dateto,flags,variables):
    """
    Read mesocale input data from netcdf file
    Inputs: 
        - file: path to netcdf file
        - datefrom: timestamp of the beginning of the evaluation period
        - dateto: timestamp of the end of the evaluation period
        - flags: list of datetimes to participate in the evaluation 
                (flags = [] will keep all data between datefrom and dateto)
        - variables: list of variables to extract from netcdf file
    Outputs: 
        - varout: as many series as variables indicated
    """
    
    f = netCDF4.Dataset(file,'r')
    times = f.variables['Times'][:]
    times = netCDF4.num2date(times,f.variables['Times'].units)
    mask = np.logical_and(times >= datefrom, times < dateto)
    idates = np.where(mask)[0]
    times= times[mask]
    varout = []
    
    for v in variables:   
        if v == 'S':
            U = pd.DataFrame(f.variables['U'][idates].data, index = times)
            V = pd.DataFrame(f.variables['V'][idates].data, index = times)
            var = (U**2 + V**2)**0.5
        elif v == 'S_corr':
            U_corr = pd.DataFrame(f.variables['U_corr'][idates].data, index = times)
            V_corr = pd.DataFrame(f.variables['V_corr'][idates].data, index = times)
            var = (U_corr**2 + V_corr**2)**0.5
        elif v == 'WD':
            U = pd.DataFrame(f.variables['U'][idates].data, index = times)
            V = pd.DataFrame(f.variables['V'][idates].data, index = times)
            var = (270-np.rad2deg(np.arctan2(V,U)))%360
        elif v == 'zL': # stability parameter zL z/L where z = 10 m
            var = 10./pd.DataFrame(f.variables['L'][idates].data, index = times)       
        else:
            var = pd.DataFrame(f.variables[v][idates].data, index = times)
        if len(flags) != 0:
            var[~var.index.duplicated()].reindex(flags).dropna()
        varout.append(var)
    if len(varout) == 1:
        varout = varout[0]
        
    return varout

def read_obs(file_obs,binmap,bins_label):
    """
    Read time-series of observed power data from csv file and return bin-averaged quantities
    Inputs: 
        - file_obs: path to csv file
        - binmap: list of timestamp indices to samples in each bin (wd,zL)
        - bins_label: list of lists of labels for each bin 
    Outputs: 
        - P_obs: xarray of bin-averaged observed power at each turbine (wt, wd, zL)
        - P_obs_std: xarray of bin standard deviation of observed power at each turbine (wt, wd, zL)
        - validation_data: flag indicating if the file exists or not
    """
 
    try:
        Nwt, Nwd, NzL = [len(dim) for dim in bins_label] 
        wt_label, WDbins_label, zLbins_label = bins_label
        
        # initialize outputs
        P_obs = xarray_init(['wt','wd','zL'],bins_label)
        P_obs_std = xarray_init(['wt','wd','zL'],bins_label)

        ts = pd.read_csv(file_obs, index_col = 'time')
        times = netCDF4.num2date(ts.index,'seconds since 1970-01-01 00:00:00.00 UTC, calendar=gregorian')
        ts['time'] = times
        ts.set_index('time',inplace=True)
        ts = ts[wt_label] 
        P_obs, P_obs_std = bin_avrg(ts,binmap,bins_label)
        validation_data = True
    except IOError:
        print ("No validation data available")
        validation_data = False
        
    return P_obs, P_obs_std, validation_data

def read_sims(sims,binmap,bins_label, printfiles = False):
    """
    Read simulation power data from csv files and return bin quantities
    The first simulation corresponds to the ensemble
    Inputs:
        - sims: dataframe with simulations description
        - binmap: list of timestamp indices to samples in each bin (wd,zL)
        - bins_label: list of lists of labels for each bin 
        - printfiles: print file names as they are loaded or not 
    Outputs: 
        - P: xarray of bin-averaged power at each turbine for all simulations (sim, wt, wd, zL)
        - Pstd: xarrary of bin standard deviation of power at each turbine for all simulations (sim, wt, wd, zL)
    """
    
    Nwt, Nwd, NzL = [len(dim) for dim in bins_label] 
    wt_label, WDbins_label, zLbins_label = bins_label
    Nsim = sims.shape[0] 
    sims_label = sims['ID'].values
    
    # initialize outputs
    P = xarray_init(['sim','wt','wd','zL'],[sims_label] + bins_label)
    P_std = xarray_init(['sim','wt','wd','zL'],[sims_label] + bins_label)

    for i_sim in range(1,Nsim):
        file_name = './outputs/'+ sims["ID"][i_sim] +'.csv'
        if printfiles:
            print(file_name)
        tsi = Pi = Pi_std = None
        if sims["Approach"][i_sim] == 'ts': # time series data
            tsi = pd.read_csv(file_name, index_col = 'time')    # read .csv output files
            times = netCDF4.num2date(tsi.index,'seconds since 1970-01-01 00:00:00.00 UTC, calendar=gregorian')
            tsi['time'] = times
            tsi.set_index('time',inplace=True)
            tsi = tsi[wt_label] 
            Pi, Pi_std = bin_avrg(tsi,binmap,bins_label)
        else: # the data already comes in bin-averaged form
            ba = pd.read_csv(file_name) # ba files are expected to come with the same order as wt_label
            Nwt = len(ba)
            bin_averages = np.empty((Nwt, Nwd, NzL))
            for i_zL in range(NzL):
                for i_wd in range(Nwd):
                    col_name = 'P' + str(i_wd+1) + zLbins_label[i_zL]
                    for i_wt in range(Nwt):
                        bin_averages[i_wt,i_wd,i_zL] = ba[col_name][i_wt]
            Pi = bin_averages
        P[i_sim] = Pi
        P_std[i_sim] = Pi_std
    
    # Compute ensemble
    ensemble = sims.index[sims['Ensemble'] == 1].tolist()
    P[0] = P[ensemble].mean(axis=0)
    P_std[0] = P_std[ensemble].mean(axis=0)
    
    if printfiles:
        print('Ensemble generated based on: ' + ', '.join(sims.loc[ensemble]["Label"].tolist()))

    return P, P_std

def spacing(x_wt, y_wt, D, WD):
    """
    Function to compute the turbine spacing in rotor diameters for a layout from a give wind direction
    Inputs:
        - x_wt, y_wt: coordinates of the wind turbines
        - D: rotor diameter
        - WD: wind direction
    Outputs: 
        - s_x/D: streemwise spacing
        - s_y/D: cross-wind spacing
    """
    n_wt = len(x_wt)
    polarangle = (-WD + 90.)*np.pi/180. # from wind direction to polar angle
    raylength = 30000.  # this should be larger than the wind farm 

    s_x = np.zeros(len(WD))
    s_y = np.zeros(len(WD))

    for wd in range(len(WD)):
        d = np.zeros((n_wt,n_wt))
        s_x_wd = np.zeros((n_wt,n_wt))
        s_y_wd = np.zeros((n_wt,n_wt))
        s_x_wd_i = np.zeros((n_wt))
        s_y_wd_i = np.zeros((n_wt))

        for i in range(n_wt):
            for j in range(n_wt):
                if i != j:
                    wt_i = Point(x_wt[i], y_wt[i])
                    wt_j = Point(x_wt[j], y_wt[j])
                    point_upstream = Point(x_wt[i] + raylength*np.cos(polarangle[wd]), 
                                           y_wt[i] + raylength*np.sin(polarangle[wd]))
                    point_downstream = Point(x_wt[i] + raylength*np.cos(polarangle[wd] + np.pi), 
                                             y_wt[i] + raylength*np.sin(polarangle[wd] + np.pi))
                    # line passing by wind turbine i and aligned with wind direction
                    line = LineString([point_upstream, point_downstream]) 
                    
                    d[i,j] = wt_i.distance(wt_j)          # distance between turbines i and j
                    s_y_wd[i,j] = line.distance(wt_j)        # cross-wind distance 
                    s_x_wd[i,j] = (d[i,j]**2 - s_y_wd[i,j]**2)**0.5  # streamwise distance

            # turbines in the same column are those with a lateral separation of less than one diameter        
            s_x_col = s_x_wd[i,s_y_wd[i,:] < D] 
            if len(s_x_col) > 1: # define streamwise spacing based on the one closest turbine in the columnt
                s_x_wd_i[i] = s_x_col[s_x_col > 0].min() 
            else:
                s_x_wd_i[i] = np.nan
            
            # turbines in the same row are those with a streamwise separation of less than one diameter
            s_y_row = s_y_wd[i,s_x_wd[i,:] < D]  
            if len(s_y_row) > 1: # define cross-wind spacing based on the one closest turbine in the row
                s_y_wd_i[i] = s_y_row[s_y_row > 0].min() 
            else:
                s_y_wd_i[i] = np.nan
        
        # wind farm spacing based on the median of the turbine spacings
        s_x[wd] = np.median(s_x_wd_i[~np.isnan(s_x_wd_i)]) 
        s_y[wd] = np.median(s_y_wd_i[~np.isnan(s_y_wd_i)])

        return s_x/D, s_y/D

def stability_metrics(data,sims,metric_label,N_WDzL,Nmin,binmap,tags, figsize = (6,6), plot = True):
    """
    Plot wind farm metrics categorized by stability for all simulations in plotresults
    Inputs: 
        - data: xarray of wind farm metric for all simulations (sim, wd, zL)
        - sims: dataframe with simulations description
        - metric_label:  name of the metric
        - N_WDzL: dataframe wind bin sample count per wd and zL (wd, zL)
        - Nmin: minimum number of samples to include the bin in the integration
        - binmap: list of timestamp indices to samples in each bin (wd,zL)
        - tags: list of simulation names 
        - figsize: figure size
        - plot: flag to return a figure
    Outputs 
        - data_zL_wide: dataframe of metric by stability for all simulations (sim, zL)
    """
    
    n_sim = len(sims)
    plotresults = sims.index[sims['Plot'] == 1].tolist()
    zLbins_label = data.coords['zL'].values.tolist()
    len_array = np.vectorize(len)
    bin_sizes = len_array(binmap)
    bin_sizes[N_WDzL < Nmin] = 0       # filter out bins with less than Nmin samples (weight = 0)
    sum_bin_sizes_zL = np.sum(bin_sizes,0)
    
    data_zL = np.sum(data*bin_sizes, axis = 1)/sum_bin_sizes_zL
    #data_zL = np.mean(data,axis=1).to_pandas() 
    data_zL_wide = data_zL.to_pandas()
    if plot:
        with sns.axes_style("darkgrid"):
            data_zL = longformat(data_zL,sims,tags,['sim','Input'],zLbins_label,metric_label)
            ax = sns.catplot(x=metric_label, y="sim", hue="Input", col="zL",
                        kind="bar", data=data_zL, palette="muted", dodge=False, height=figsize[0], aspect=figsize[1]);
            ax.set_axis_labels(metric_label, "")
            #annotate_bars(ax,data_zL,metric)            
    return data_zL_wide
    
def time_stamp(year,month,day,hour,minute,second):
    """ Abstraction method for creating compatible time stamps"""
    return datetime.datetime(year,month,day,hour,minute,second)   

def WDzL_bins(x,y,ts,statistic,bins,bins_label,plot = False):
    """
    Compute and plot distribution of samples per bin
    Inputs:
        - x: time-series of wind direction
        - y: time-series of stability
        - ts: time-series of values
        - statistic: which statistic to compute per bin
        - bins: bin limits for x and y
        - bins_label: bin labels
        - plot: whether to plot the distribution or not
    Outputs: 
        - N_WDzL: dataframe wind bin sample count per wd and zL
        - binmap: list of timestamp indices to samples in each bin (wd,zL)
    """
    
    Nwt, Nwd, NzL = [len(dim) for dim in bins_label]
    wt_label, WDbins_label, zLbins_label = bins_label
    WDbins, zLbins = bins
    x = x.values.flatten()
    x[x>WDbins[-1]] = x[x>WDbins[-1]]-360
    y = y.values.flatten()
    statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, ts.values.flatten(), 
                                                                     statistic=statistic, 
                                                                     bins=bins, expand_binnumbers = True)
    N_WDzL = pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label)

    binmap = np.empty((Nwd, NzL), dtype = object)
    for i_wd in range(Nwd):
        for i_zL in range(NzL):
            binmap[i_wd,i_zL] = ts[np.logical_and(binnumber[0,:] == i_wd+1, binnumber[1,:] == i_zL+1)].index

    if plot:
        N_zL = np.sum(N_WDzL, axis = 0).rename('pdf')
        N_WD = np.sum(N_WDzL, axis = 1).rename('pdf')
        Nnorm_WDzL = N_WDzL.div(N_WD, axis=0)
        NzL = len(bins_label[2])

        f1 = plt.figure(figsize = (18,8))
        cmap = plt.get_cmap('bwr')
        zLcolors = np.flipud(cmap(np.linspace(0.,NzL,NzL)/NzL))
        ax1=Nnorm_WDzL.plot.bar(stacked=True, color=zLcolors, align='center', width=1.0, legend=False, 
                                rot=90, use_index = False, edgecolor='grey')
        ax2=(N_WD/N_WD.sum()).plot(ax=ax1, secondary_y=True, style='k',legend=False, rot=90, use_index = False);
        ax2.set_xticklabels(WDbins_label)
        #ax1.set_title('Wind direction vs stability')
        ax1.set_ylabel('$pdf_{norm}$($z/L_{ref}$)')
        ax2.set_ylabel('$pdf$($WD_{ref}$)', rotation=-90, labelpad=15)
        ax1.set_yticks(np.linspace(0,1.,6))
        ax1.set_ylim([0,1.])
        ax2.set_yticks(np.linspace(0,0.2,6))

        h1, l1 = ax1.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        plt.legend(h1+h2, l1+l2, bbox_to_anchor=(1.3, 1))

        #cellText = N_WDzL.T.astype(str).values.tolist() # add table to bar plot
        #the_table = plt.table(cellText=cellText,
        #                      rowLabels=zLbins_label,
        #                      rowColours=zLcolors,
        #                      colLabels=WDbins_label,
        #                      loc='bottom')

    return N_WDzL, binmap

def windfarmlength(points, wd):
    """
    Compute wind farm length for a given wind direction
    Inputs:
        - points: (x,y) turbine coordinates in meters
        - wd:  wind direction in degress
    Outputs:
        - length: wind farm length in meters
    """
    raylength = 30000 
    alpha = 0.0004
    alpha_shape = alphashape.alphashape(points, alpha)
    polarangle = (-wd + 90.)*np.pi/180. # from wind direction to polar angle    
    length = []
    
    for i in polarangle:
        point_upstream = Point(raylength*np.cos(i), raylength*np.sin(i))
        point_downstream = Point(raylength*np.cos(i + np.pi), raylength*np.sin(i + np.pi))
        line = LineString([point_upstream, point_downstream])
        x = alpha_shape.boundary.intersection(line) # intersection between line and boundary of wind farm
        length.append(x[0].distance(x[1]))
    length = np.asarray(length)
    
    return length

def xarray_init(dims,labels):
    """
    Initialize xarray
    Inputs: 
        - dims: list with dimmension names
        - labels: coordinates of each dimmension (bins labels)
    outputs: 
        - xr_init: empty xarray 
    """
    Nbin = [len(dim) for dim in labels]
    dd = dict.fromkeys(dims, None)
    for d in range(len(dims)):
        dd[dims[d]] = labels[d]
    xr_init = xr.DataArray(np.empty(Nbin), coords = dd, dims = dims)   
    return xr_init

