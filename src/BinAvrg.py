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

def plot_wf_layout(x, y, data, labels = [], vmin = np.nan, vmax = np.nan, cmap = 'jet', title = [], figsize=(12,6),ax = None):  
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    if len(data) == 0:
        sc = ax.scatter(x,y,marker='o',c=data,edgecolors ='k')
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

def centroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def flags_to_ts(scada_flags, min_data_availability):
    filter_condition = (100 - min_data_availability)/100 * scada_flags.shape[1]
    flags_filter = (scada_flags.sum(axis=1) < filter_condition) 
    return flags_filter[flags_filter].keys()

def longformat(data,sims,tags,id_vars,value_vars,metric_label):
    plotresults = sims.index[sims['Plot'] == 1].tolist()
    data = data[plotresults].to_pandas()
    data['sim'] = sims.loc[plotresults][tags].values
    data['Input'] = sims.loc[plotresults]['Input'].values
    data['Model'] = sims.loc[plotresults]['Model Type'].values
    for i_sim in range(len(plotresults)):
        data.loc[sims['ID'][plotresults[i_sim]],'K'] = sims.loc[plotresults[i_sim]]['Remarks'].partition('K = ')[2]    
    data = pd.melt(data, id_vars=id_vars, value_vars=value_vars)
    data.rename(columns={'value':metric_label}, inplace=True)  
    return data

def plot_eta_WD_zL(sim_eta, bias, bench_eta, sims, N_WDzL, validation):   
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        plotresults = sims.index[sims['Plot'] == 1].tolist() 
        tags = 'Label' 

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

        eta_zL = longformat(eta_zL,sims,tags,['sim','Input','Model'],zLbins_label,"ArrayEff[%]")
        bias_zL = longformat(bias_zL,sims,tags,['sim','Input','Model'],zLbins_label,"BIAS[%]")
        eta_WD = longformat(eta_WD,sims,tags,['sim','Input','Model'],WDbins_label,"ArrayEff[%]")
        bias_WD = longformat(bias_WD,sims,tags,['sim','Input','Model'],WDbins_label,"BIAS[%]")

        f1 = plt.figure(figsize = (14,8), constrained_layout = True)   
        spec = gridspec.GridSpec(ncols=2, nrows=2, figure=f1, width_ratios=[2, 0.6], height_ratios=[2, 1])    

        f1_ax1 = f1.add_subplot(spec[0, 0])
        if validation:
            eta_obs_WD = bench_eta.mean(axis = 1, skipna = True).to_pandas()
            eta_obs_WD.plot(grid=1, marker='s', markerfacecolor='grey', linewidth = 3, markeredgecolor= 'k', 
                            markersize = 8, color = 'k', linestyle='--', label = 'Observations', ax = f1_ax1)
        f1_ax1 = sns.lineplot(x='wd', y='ArrayEff[%]', hue='Input', style = 'Model', data=eta_WD, 
                              sort=False, legend=False, ax = f1_ax1)
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
        if validation:
            eta_obs_zL = bench_eta.mean(axis = 0, skipna = True).to_pandas()
            eta_obs_zL.plot(grid=1, marker='s', markerfacecolor='grey', linewidth = 2, markeredgecolor= 'k', 
                            markersize = 8, color = 'k', linestyle='--', label = 'Observations', ax = f1_ax3)
        f1_ax3 = sns.lineplot(x='zL', y='ArrayEff[%]', hue='Input', style = 'Model', data=eta_zL, 
                              sort=False, ax = f1_ax3)
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

def plot_transect(data,val_data,meso_data,obsPfree_data,wt_list,turbines,Drot,sims,plotresults,WDbin,zLbin,highlight,wtnorm,validation, ylim1 = [0.6,1.4], ylim2 = [0.9,1.1], figsize = (16,8)):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        # Normalize based on wtnorm position
        if wtnorm == 'ref':
            datanorm = data.loc[:,wt_list]/data.mean(axis=1)
            valdatanorm = val_data.loc[wt_list]/val_data.mean()
            #datanorm = data[wt_list].div(data.mean(axis = 1), axis=0)
            #valdatanorm = val_data[wt_list].div(val_data.mean())
        else:
            datanorm = data.loc[:,wt_list]/data.loc[:,wtnorm]
            valdatanorm = val_data.loc[wt_list]/val_data.loc[wtnorm]
            #datanorm = data[wt_list].div(data[wtnorm], axis=0)
            #valdatanorm = val_data[wt_list].div(val_data[wtnorm].mean())

        tags = 'Label'
        datanorm_sns = longformat(datanorm,sims,tags,['sim','Input','Model'],wt_list,"P")

        n_wt = len(wt_list)

        #compute distances to first turbine
        a = turbines.loc[turbines['VDC ID'] == wt_list[0],['X coordinate','Y coordinate']].values.flatten()            
        dists = []
        coords = np.zeros((n_wt,2))
        for wt in range(n_wt):
            b = turbines.loc[turbines['VDC ID'] == wt_list[wt],['X coordinate','Y coordinate']].values.flatten()
            dists.append(((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5 / Drot)
            coords[wt,:]=b

        f1 = plt.figure(figsize = figsize)   # constrained_layout=True
        spec = gridspec.GridSpec(ncols=2, nrows=2, figure=f1, width_ratios=[1, 1.5], height_ratios=[2, 1])    

        iwt = [x in wt_list for x in turbines['VDC ID']]
        f1_ax1 = f1.add_subplot(spec[:, 0])
        f1_ax1.scatter(turbines['X coordinate'],turbines['Y coordinate'],c='silver', marker = 'o', s=20)
        f1_ax1.scatter(turbines['X coordinate'][iwt],turbines['Y coordinate'][iwt],c='black', marker = 'o',s=20)
        f1_ax1.text(coords[0][0],coords[0][1],wt_list[0],{'ha': 'right'}, fontsize=14)
        f1_ax1.text(coords[-1][0],coords[-1][1],wt_list[-1],{'ha': 'right'}, fontsize=14)
        f1_ax1.axis('scaled')
        f1_ax1.spines['top'].set_visible(False)
        f1_ax1.spines['bottom'].set_visible(False)
        f1_ax1.spines['left'].set_visible(False)
        f1_ax1.spines['right'].set_visible(False)
        f1_ax1.get_xaxis().set_ticks([])
        f1_ax1.get_yaxis().set_ticks([])

        f1_ax2 = f1.add_subplot(spec[0, 1])
        if validation:
            valdatanorm.to_pandas().plot(marker='s', markerfacecolor='grey', linewidth = 2, markeredgecolor= 'k', 
                             markersize = 8, color = 'k', linestyle='--', ax = f1_ax2, label = 'Observations')
        f1_ax2 = sns.lineplot(x='wt', y='P', hue='Input', style = 'Model', data=datanorm_sns, sort=False) 
        f1_ax2.set_xticklabels(wt_list)
        f1_ax2.legend(loc = 'upper left', bbox_to_anchor=(1, 1)) 
        f1_ax2.set_ylim(ylim1)
        f1_ax2.set_xlabel(None)
        if wtnorm == 'ref':
            f1_ax2.set_ylabel('Net Power Ratio = $P/P_{ref}$')  
        else:
            f1_ax2.set_ylabel('Net Power Ratio = P/P$_{}$'.format(wtnorm))  
        f1_ax2.set_title('Transect '+wt_list[0]+'-'+wt_list[-1]+' ('+WDbin+', '+zLbin+')')
        f1_ax2.grid(True)
        f1_ax2.margins(0.05)

        if wtnorm == 'ref':
            meso_P_ratio = meso_data.loc[wt_list]/meso_data.mean()
        else:
            meso_P_ratio = meso_data.loc[wt_list]/meso_data.loc[wtnorm]

        if validation: 
            if wtnorm == 'ref':
                obs_Pfree_ratio = obsPfree_data.loc[wt_list]/obsPfree_data.mean()
            else:
                obs_Pfree_ratio = obsPfree_data.loc[wt_list]/obsPfree_data.loc[wtnorm]

        f1_ax3 = f1.add_subplot(spec[1, 1])
        f1_ax3.plot(wt_list,meso_P_ratio,'-b',linewidth = 2,label = 'Meso')
        if validation:
            f1_ax3.plot(wt_list,obs_Pfree_ratio,marker='s', markerfacecolor='grey', linewidth = 2,
                        markeredgecolor= 'k',  markersize = 8, color = 'k', linestyle='--', label = 'Obs')
        if wtnorm == 'ref':
            f1_ax3.set_ylabel('Gross Power Ratio = $P(S)/P(S_{ref})$') 
        else:
            f1_ax3.set_ylabel('Gross Power Ratio = P(S)/P(S$_{}$)'.format(wtnorm))    

        f1_ax3.legend() #loc = 'upper left', bbox_to_anchor=(1, 1)
        f1_ax3.grid(True)
        f1_ax3.set_ylim(ylim2)

        xlim = f1_ax3.get_xlim()
        f1_ax2.set_xlim(xlim)

    return f1_ax1,f1_ax2,f1_ax3

def read_input(file,datefrom,dateto,flags,variables):
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
        elif v == 'zL':
            var = 10./pd.DataFrame(f.variables['L'][idates].data, index = times)
        else:
            var = pd.DataFrame(f.variables[v][idates].data, index = times)
        if len(flags) != 0:
            var[~var.index.duplicated()].reindex(flags).dropna()
        varout.append(var)
    if len(varout) == 1:
        varout = varout[0]
    return varout

def WDzL_bins(x,y,ts,statistic,bins,bins_label,plot = False):
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

        f1 = plt.figure(figsize = (18,8));
        cmap = plt.get_cmap('bwr');
        zLcolors = np.flipud(cmap(np.linspace(0.,NzL,NzL)/NzL));
        ax1=Nnorm_WDzL.plot.bar(stacked=True, color=zLcolors, align='center', width=1.0, legend=False, 
                                rot=90, use_index = False, edgecolor='grey');
        ax2=(N_WD/N_WD.sum()).plot(ax=ax1, secondary_y=True, style='k',legend=False, rot=90, use_index = False);
        ax2.set_xticklabels(WDbins_label);
        #ax1.set_title('Wind direction vs stability');
        ax1.set_ylabel('$pdf_{norm}$($z/L_{ref}$)');
        ax2.set_ylabel('$pdf$($WD_{ref}$)', rotation=-90, labelpad=15);
        ax1.set_yticks(np.linspace(0,1.,6));
        ax1.set_ylim([0,1.]);
        ax2.set_yticks(np.linspace(0,0.2,6));

        h1, l1 = ax1.get_legend_handles_labels();
        h2, l2 = ax2.get_legend_handles_labels();
        plt.legend(h1+h2, l1+l2, bbox_to_anchor=(1.3, 1));
        #plt.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.2)  


        #cellText = N_WDzL.T.astype(str).values.tolist()
        #the_table = plt.table(cellText=cellText,
        #                      rowLabels=zLbins_label,
        #                      rowColours=zLcolors,
        #                      colLabels=WDbins_label,
        #                      loc='bottom')

    return N_WDzL, binmap

def xarray_init(dims,labels):
    Nbin = [len(dim) for dim in labels]
    dd = dict.fromkeys(dims, None)
    for d in range(len(dims)):
        dd[dims[d]] = labels[d]
    xr_init = xr.DataArray(np.empty(Nbin), coords = dd, dims = dims)   
    return xr_init

def bin_avrg(ts,binmap,bins_label):
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

def read_sims(sims,binmap,bins_label, printfiles = False):
    Nwt, Nwd, NzL = [len(dim) for dim in bins_label] 
    wt_label, WDbins_label, zLbins_label = bins_label
    Nsim = sims.shape[0] 
    sims_label = sims['ID'].values
        
    P = xarray_init(['sim','wt','wd','zL'],[sims_label] + bins_label)
    P_std = xarray_init(['sim','wt','wd','zL'],[sims_label] + bins_label)

    for i_sim in range(1,Nsim):
        file_name = './outputs/'+ sims["ID"][i_sim] +'.csv'
        if printfiles:
            print(file_name)
        tsi = Pi = Pi_std = None
        if sims["Approach"][i_sim] == 'ts':
            tsi = pd.read_csv(file_name, index_col = 'time')    # read .csv output files
            times = netCDF4.num2date(tsi.index,'seconds since 1970-01-01 00:00:00.00 UTC, calendar=gregorian')
            tsi['time'] = times
            tsi.set_index('time',inplace=True)
            tsi = tsi[wt_label] 
            Pi, Pi_std = bin_avrg(tsi,binmap,bins_label)
        else:
            ba = pd.read_csv(file_name) # ba files are expected to come with the same order as wt_label
            Nwt = len(ba)
            bin_averages = np.empty((Nwt, Nwd, NzL))
            for i_zL in range(NzL):
                for i_wd in range(Nwd):
                    col_name = 'P' + str(i_wd+1) + zLbins_label[i_zL]
                    for i_wt in range(Nwt):
                        bin_averages[i_wt,i_wd,i_zL] = ba[col_name][i_wt]
        P[i_sim] = Pi
        P_std[i_sim] = Pi_std
    # Compute ensemble
    ensemble = sims.index[sims['Ensemble'] == 1].tolist()
    P[0] = P[ensemble].mean(axis=0)
    P_std[0] = P_std[ensemble].mean(axis=0)
    if printfiles:
        print('Ensemble generated based on: ' + ', '.join(sims.loc[ensemble]["Label"].tolist()))

    return P, P_std

def read_obs(file_obs,binmap,bins_label):
    try:
        Nwt, Nwd, NzL = [len(dim) for dim in bins_label] 
        wt_label, WDbins_label, zLbins_label = bins_label

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


def define_benchmark(validation, validation_data, benchmark,P_obs,P_obs_std,P,P_std, sims):
    if (validation and validation_data):
        print ("You are doing validation against observations")
        P_bench = P_obs
        P_bench_std = P_obs_std
    else: 
        print ("You are doing code-to-code assessment against the benchmark series $%s$ " % (sims["Label"][benchmark]))
        P_bench = P[benchmark]
        P_bench_std = P_std[benchmark]
    return P_bench, P_bench_std

def read_Am(file_obs,N_WDzL,binmap,bins_label,P_meso,file_Am,savefile = False, plot = True):
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
            fig, ax = plt.subplots(figsize=(10,6))
            ax = sns.barplot(x='wd', y='Am', hue='zL', data=Am_pd, palette = 'bwr_r',
                             edgecolor='grey',saturation=1) #sns.color_palette("coolwarm", 3)
            ax.grid(True)
            ax.set_title('Wind farm mesoscale bias correction factor $A_M$')
            ax.legend(loc = 'upper left', bbox_to_anchor=(1, 1));
            sns.axes_style("white")
    return Am, Pfree_obs, Pfree_obs_std

def Afreestream(P_gross,P,sims,bins,bins_label,turbines):
    P_mesowk,P_mesoctrl,P_mesocorr,Pfree_obs = P_gross

    WDbins, zLbins = bins
    Nwt, Nwd, NzL = [len(dim) for dim in bins_label] 
    wt_label, WDbins_label, zLbins_label = bins_label
    Nsim = sims.shape[0] 
    sims_label = sims['ID'].values

    Pfree = xarray_init(['sim','wt','wd','zL'],[sims_label] + bins_label)
    Afree = xarray_init(['sim','wt','wd','zL'],[sims_label] + bins_label)

    points = turbines[['X coordinate','Y coordinate']].values
    pbar = tqdm(total = Nsim, leave = False, position = 0, desc="Progress")
    for i_sim in range(Nsim):
        for i_wd in range(Nwd):
            for i_zL in range(NzL):
                WD = 0.5*(WDbins[i_wd] + WDbins[i_wd+1])
                freestream = define_freestream(points,WD)
                f1 = interpolate.NearestNDInterpolator(points[freestream], P[i_sim][freestream,i_wd,i_zL].values)
                Pfree[i_sim][:,i_wd,i_zL] = f1(points)
        if ("wakes-corr" in sims['Input'][i_sim]):
            Afree[i_sim] = Pfree[i_sim]/P_mesocorr
        elif ("ctrl" in sims['Input'][i_sim]):
            Afree[i_sim] = Pfree[i_sim]/P_mesoctrl
        elif ("scada" in sims['Input'][i_sim]):
            Afree[i_sim] = Pfree[i_sim]/Pfree_obs
        else:
            Afree[i_sim] = Pfree[i_sim]/P_mesowk     
        pbar.update(1)
    pbar.close()
        
    return Pfree, Afree

def plot_Afree(Afree,sim,N_WDzL,zLbins_label):
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
        ax.legend(loc = 'upper left', bbox_to_anchor=(1, 1));
        sns.axes_style("white")
        
    Afree_global = Afree.loc[sim].mean(axis=0).drop('sim')
    Afree_global_df = Afree_global.to_dataframe(name = 'Afree').unstack()
    Afree_global_df.columns = Afree_global_df.columns.get_level_values(1)
    Afree_global_df = Afree_global_df[zLbins_label]

    Afree_total = (Afree_global_df*N_WDzL/N_WDzL.sum().sum()).sum().sum()
    print("Total freestream bias correction = %.3f" % (Afree_total))
    
    return ax
   
def annotate_bars(ax,data,col):
    cnt = 0
    for index, row in data.iterrows():
        if row[col]<0: 
            ha = 'left'
        else:
            ha = 'right'
        ax.text(row[col],cnt+0.25, round(row[col],2), color='k', ha=ha)
        cnt = cnt + 1

def compute_overall_metrics(eta_farm,bias_farm,mae_farm,sims,N_WDzL,Nmin,binmap,tags,figsize = (10,6), plot = True):   
    n_sim = len(sims)
    plotresults = sims.index[sims['Plot'] == 1].tolist()
    len_array = np.vectorize(len)
    bin_sizes = len_array(binmap)
    bin_sizes[N_WDzL < Nmin] = 0 
    sum_bin_sizes = bin_sizes.sum()

    eta_farm_tot = np.sum(eta_farm*bin_sizes/sum_bin_sizes, axis = (1,2)) 
    bias_farm_tot = np.sum(bias_farm*bin_sizes/sum_bin_sizes, axis = (1,2)) 
    mae_farm_tot = np.sum(mae_farm*bin_sizes/sum_bin_sizes, axis = (1,2)) 

    # if there are no result for all bins don't compute the total values
    notallbins = np.any(np.isnan(eta_farm), axis = (1,2))
    bias_farm_tot[notallbins] = np.nan
    mae_farm_tot[notallbins] = np.nan
    eta_farm_tot[notallbins] = np.nan

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
            data = metrics.iloc[plotresults]
            ax[0] = sns.barplot(x="ArrayEff[%]", y="sim", hue="Input", data=data,
                        palette="muted", dodge=False, ax = ax[0])
            ax[0].legend([])
            ax[0].set_ylabel('')
            ax[0].grid(True)
            annotate_bars(ax[0],data,"ArrayEff[%]")

            ax[1] = sns.barplot(x="BIAS[%]", y="sim", hue="Input", data=data,
                        palette="muted", dodge=False, ax = ax[1])
            ax[1].legend([])
            ax[1].grid(True)
            ax[1].set_ylabel('')    
            annotate_bars(ax[1],data,"BIAS[%]")

            ax[2] = sns.barplot(x="MAE[%]", y="sim", hue="Input", data=data,
                        palette="muted", dodge=False, ax = ax[2])
            ax[2].legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax[2].grid(True)
            ax[2].set_ylabel('')
            annotate_bars(ax[2],data,"MAE[%]")
                
                
        sns.axes_style("white")
    return metrics


def compute_stability_metrics(data,sims,metric_label,N_WDzL,Nmin,binmap,tags, figsize = (6,6), plot = True):
    n_sim = len(sims)
    plotresults = sims.index[sims['Plot'] == 1].tolist()
    zLbins_label = data.coords['zL'].values.tolist()
    len_array = np.vectorize(len)
    bin_sizes = len_array(binmap)
    bin_sizes[N_WDzL < Nmin] = 0 
    sum_bin_sizes_zL = np.sum(bin_sizes,0)
    
    data_zL = np.sum(data*bin_sizes, axis = 1)/sum_bin_sizes_zL
    #data_zL = np.mean(data,axis=1).to_pandas() 
    data_zL_wide = data_zL.to_pandas()
    if plot:
        with sns.axes_style("darkgrid"):
            data_zL = longformat(data_zL,sims,tags,['sim','Input'],zLbins_label,metric_label)
            ax = sns.catplot(x=metric_label, y="sim", hue="Input", col="zL",
                        kind="bar", data=data_zL, palette="muted", dodge=False);
            ax.set_axis_labels(metric_label, "")
            #annotate_bars(ax,data_zL,metric)            
    return data_zL_wide
    

def plot_heatmaps(data,sims, N_WDzL, Nmin, ax_size = (1.7,4), cols = 4):
    plotresults = sims.index[sims['Plot'] == 1].tolist()
    data = data[plotresults]
    mask = np.zeros_like(N_WDzL)
    mask[N_WDzL > Nmin] = True

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
                        cmap="bwr_r",cbar = False, mask = mask)
            ax[index].set_title(sim_id[iax])
            ax[index].set_xlabel('')
            ax[index].set_ylabel('')
        plt.tight_layout()
    sns.axes_style("white");    
        
def plot_layout_timestamp():
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