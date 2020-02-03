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

def plot_wf_layout(x,y,labels = [],figsize=(12,6), data = [],vmin = np.nan,vmax = np.nan):
    if len(data)==0:
        fig, ax = plt.subplots(figsize=figsize)
        plt.scatter(x,y)
        #labeled wind turbines    
        for i in range(len(labels)):
            plt.text(x[i]+0.4, y[i]+0.4, labels[i], fontsize=9) 
        ax.set_aspect(1.0)
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.grid()
        ax.axis('equal')
        ax.axis([min(x)-5000, max(x)+2000,min(y)-2000, max(y)+2000])
    else:
        fig,ax = plt.subplots(1, 2, figsize=figsize)
        for iax in range(len(data)):
            if vmin[iax]<0:
                cmap=plt.cm.get_cmap('bwr',16)
                ax[iax].set_title('BIAS[%]')
            else:
                cmap=plt.cm.get_cmap('jet',10)
                ax[iax].set_title('Array Eff.[%]')
            sc = ax[iax].scatter(x,y,marker='o',c=data[iax],cmap=cmap,edgecolors ='k',vmin=vmin[iax], vmax=vmax[iax])
            plt.colorbar(sc,ax=ax[iax])
            ax[iax].set_aspect(1.0)
            ax[iax].set_xlabel('X [m]')
            ax[iax].set_ylabel('Y [m]')
            ax[iax].grid()

    plt.show()

def centroid(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length

def plot_eta_WD_zL(sims, plotresults, sim_eta, bias, bench_eta, N_WDzL_speed, validation):   
       
    eta_zL = sim_eta.mean(axis = 1, skipna = True).to_pandas() # array efficiency vs wind direction 
    bias_zL = bias.mean(axis = 1, skipna = True).to_pandas()   # bias vs wind direction [%] 
    N_zL = N_WDzL_speed.sum(axis = 0, skipna = True) # sample count

    eta_WD = sim_eta.mean(axis = 2, skipna = True).to_pandas() # array efficiency vs wind direction 
    bias_WD = bias.mean(axis = 2, skipna = True).to_pandas()   # bias vs wind direction [%] 
    N_WD = N_WDzL_speed.sum(axis = 1, skipna = True) # sample count
    
    zLbins_label = sim_eta.coords['zL'].values.tolist()
    WDbins_label = sim_eta.coords['wd'].values.tolist()
    
    eta_zL = eta_zL.loc[sims['ID'][plotresults]] # plot only these simulations
    eta_zL.reset_index(level=0, inplace=True)
    eta_zL = pd.melt(eta_zL, id_vars=['sim'], value_vars=zLbins_label)
    eta_zL.rename(columns={'value':'eta'}, inplace=True)
    eta_zL['Input'] = 'wakes' # initialize
    eta_zL['K'] = 0           # initialize
    eta_zL['Model'] = 'model' # initialize 
    for isim in range(len(plotresults)):
        eta_zL.loc[eta_zL['sim']==sims['ID'][plotresults[isim]],'Input'] = sims['Input'][plotresults[isim]]
        eta_zL.loc[eta_zL['sim']==sims['ID'][plotresults[isim]],'K'] = sims['Remarks'][plotresults[isim]].partition('K = ')[2]
        eta_zL.loc[eta_zL['sim']==sims['ID'][plotresults[isim]],'Model'] = sims['Model Type'][plotresults[isim]]

    bias_zL = bias_zL.loc[sims['ID'][plotresults]] # plot only these simulations
    bias_zL.reset_index(level=0, inplace=True)
    bias_zL = pd.melt(bias_zL, id_vars=['sim'], value_vars=zLbins_label)
    bias_zL.rename(columns={'value':'bias'}, inplace=True)
    bias_zL['Input'] = 'wakes' # initialize
    bias_zL['K'] = 0           # initialize
    bias_zL['Model'] = 'model' # initialize 
    for isim in range(len(plotresults)):
        bias_zL.loc[eta_zL['sim']==sims['ID'][plotresults[isim]],'Input'] = sims['Input'][plotresults[isim]]
        bias_zL.loc[eta_zL['sim']==sims['ID'][plotresults[isim]],'K'] = sims['Remarks'][plotresults[isim]].partition('K = ')[2]
        bias_zL.loc[eta_zL['sim']==sims['ID'][plotresults[isim]],'Model'] = sims['Model Type'][plotresults[isim]]

    eta_WD = eta_WD.loc[sims['ID'][plotresults]] # plot only these simulations
    eta_WD.reset_index(level=0, inplace=True)
    eta_WD = pd.melt(eta_WD, id_vars=['sim'], value_vars=WDbins_label)
    eta_WD.rename(columns={'value':'eta'}, inplace=True)
    eta_WD['Input'] = 'wakes' # initialize
    eta_WD['K'] = 0           # initialize
    eta_WD['Model'] = 'model' # initialize 
    for isim in range(len(plotresults)):
        eta_WD.loc[eta_WD['sim']==sims['ID'][plotresults[isim]],'Input'] = sims['Input'][plotresults[isim]]
        eta_WD.loc[eta_WD['sim']==sims['ID'][plotresults[isim]],'K'] = sims['Remarks'][plotresults[isim]].partition('K = ')[2]
        eta_WD.loc[eta_WD['sim']==sims['ID'][plotresults[isim]],'Model'] = sims['Model Type'][plotresults[isim]]

    bias_WD = bias_WD.loc[sims['ID'][plotresults]] # plot only these simulations
    bias_WD.reset_index(level=0, inplace=True)
    bias_WD = pd.melt(bias_WD, id_vars=['sim'], value_vars=WDbins_label)
    bias_WD.rename(columns={'value':'bias'}, inplace=True)
    bias_WD['Input'] = 'wakes' # initialize
    bias_WD['K'] = 0           # initialize
    bias_WD['Model'] = 'model' # initialize 
    for isim in range(len(plotresults)):
        bias_WD.loc[eta_WD['sim']==sims['ID'][plotresults[isim]],'Input'] = sims['Input'][plotresults[isim]]
        bias_WD.loc[eta_WD['sim']==sims['ID'][plotresults[isim]],'K'] = sims['Remarks'][plotresults[isim]].partition('K = ')[2]
        bias_WD.loc[eta_WD['sim']==sims['ID'][plotresults[isim]],'Model'] = sims['Model Type'][plotresults[isim]]

    f1 = plt.figure(figsize = (14,8), constrained_layout = True)   
    spec = gridspec.GridSpec(ncols=2, nrows=2, figure=f1, width_ratios=[2, 0.6], height_ratios=[2, 1])    

    f1_ax1 = f1.add_subplot(spec[0, 0])
    if validation:
        eta_obs_WD = bench_eta.mean(axis = 1, skipna = True).to_pandas()
        eta_obs_WD.plot(grid=1, marker='s', markerfacecolor='grey', linewidth = 3, markeredgecolor= 'k', 
                        markersize = 8, color = 'k', linestyle='--', label = 'Observations', ax = f1_ax1)
    f1_ax1 = sns.lineplot(x='wd', y='eta', hue='Input', style = 'Model', data=eta_WD, 
                          sort=False, legend=False, ax = f1_ax1)
    f1_ax1.set_xticklabels(WDbins_label)
    f1_ax1.set_ylabel('Array Efficiency [%]')
    f1_ax1.grid(True)
    #f1_ax1.get_legend().remove()
    f1_ax1.set_ylim([40,100])
    f1_ax1b=f1_ax1.twinx()
    N_WD.plot.bar(color = 'silver', ax = f1_ax1b)
    f1_ax1b.set_yticks([0,250,500,750,1000])
    f1_ax1b.set_ylim([0,5000])

    f1_ax2 = f1.add_subplot(spec[1, 0], sharex=f1_ax1)
    f1_ax2 = sns.barplot(x='wd', y='bias', hue='Input', data=bias_WD, ax = f1_ax2)
    f1_ax2.set_ylabel('BIAS [%]')
    f1_ax2.get_legend().remove()
    f1_ax2.set_ylim([-20,20])
    f1_ax2.grid(True)

    f1_ax3 = f1.add_subplot(spec[0, 1], sharey=f1_ax1)
    if validation:
        eta_obs_zL = bench_eta.mean(axis = 0, skipna = True).to_pandas()
        eta_obs_zL.plot(grid=1, marker='s', markerfacecolor='grey', linewidth = 2, markeredgecolor= 'k', 
                        markersize = 8, color = 'k', linestyle='--', label = 'Observations', ax = f1_ax3)
    f1_ax3 = sns.lineplot(x='zL', y='eta', hue='Input', style = 'Model', data=eta_zL, 
                          sort=False, ax = f1_ax3)
    f1_ax3.grid(True)
    f1_ax3.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
    f1_ax3b=f1_ax3.twinx()
    N_zL.plot.bar(color = 'silver', ax = f1_ax3b)
    f1_ax3b.set_yticks([0,1000,2000,3000,4000])
    f1_ax3b.set_ylim([0,8000])
    f1_ax3b.set_ylabel('Samples')
    f1_ax3b.yaxis.set_label_coords(1.2,0.1)
    f1_ax3.set_xticklabels('')
    f1_ax3.set_xlabel('')

    f1_ax4 = f1.add_subplot(spec[1, 1], sharex=f1_ax3, sharey=f1_ax2)
    f1_ax4 = sns.barplot(x='zL', y='bias', hue='Input', data=bias_zL, ax = f1_ax4)
    f1_ax4.legend(loc = 'upper left', bbox_to_anchor=(1, 1))
    #f1_ax4.set_yticklabels('')
    f1_ax4.set_ylabel('')
    f1_ax4.grid(True)

    return f1_ax1, f1_ax2, f1_ax3, f1_ax4

def plot_transect(data,val_data,meso_data,obsPfree_data,wt_list,turbines,Drot,sims,plotresults,WDbin,zLbin,highlight,ylim1,ylim2,figsize,wtnorm,validation):
    # Normalize based on wtnorm position
    if wtnorm == 'ref':
        datanorm = data[wt_list].div(data.mean(axis = 1), axis=0)
        valdatanorm = val_data[wt_list].div(val_data.mean())
    else:
        datanorm = data[wt_list].div(data[wtnorm], axis=0)
        valdatanorm = val_data[wt_list].div(val_data[wtnorm].mean())

    datanorm = datanorm.loc[sims['ID'][plotresults]] # plot only these simulations
    datanorm.reset_index(level=0, inplace=True)
    datanorm_sns = pd.melt(datanorm, id_vars=['sim'], value_vars=wt_list)
    datanorm_sns.rename(columns={'value':'P'}, inplace=True)
    datanorm_sns['Input'] = 'wakes' # initialize
    datanorm_sns['K'] = 0           # initialize
    datanorm_sns['Model'] = 'model' # initialize 
    for isim in range(len(plotresults)):
        datanorm_sns.loc[datanorm_sns['sim']==sims['ID'][plotresults[isim]],'Input'] = sims['Input'][plotresults[isim]]
        datanorm_sns.loc[datanorm_sns['sim']==sims['ID'][plotresults[isim]],'K'] = sims['Remarks'][plotresults[isim]].partition('K = ')[2]
        datanorm_sns.loc[datanorm_sns['sim']==sims['ID'][plotresults[isim]],'Model'] = sims['Model Type'][plotresults[isim]]

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
        valdatanorm.plot(marker='s', markerfacecolor='grey', linewidth = 2, markeredgecolor= 'k', 
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

    meso_P_ratio = meso_data.reindex(wt_list)
    if wtnorm == 'ref':
        meso_P0 = meso_data.mean()
    else:
        meso_P0 = meso_data.loc[wtnorm]
    meso_P_ratio = meso_P_ratio/meso_P0

    if validation: 
        obs_Pfree_ratio = obsPfree_data.reindex(wt_list)
        if wtnorm == 'ref':
            obs_Pfree_P0 = obsPfree_data.mean()
        else:
            obs_Pfree_P0 = obsPfree_data.loc[wtnorm]
        obs_Pfree_ratio = obs_Pfree_ratio/obs_Pfree_P0

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

def plot_transect_old(data,ref_data,meso_data,wt_list,turbines,rot_d,sim_name,WDbin,zLbin):
    n_wt = len(wt_list)
    
    #compute distances
    a = turbines.loc[turbines['VDC ID'] == wt_list[0],['X coordinate','Y coordinate']].values.flatten()
    dists = []
    coords = np.zeros((n_wt,2))
    for wt in range(n_wt):
        b = turbines.loc[turbines['VDC ID'] == wt_list[wt],['X coordinate','Y coordinate']].values.flatten()
        dists.append(((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5 / rot_d)
        coords[wt,:]=b
        
    ref_data = ref_data.reindex(wt_list)
#    ref_data_std = ref_data_std.reindex(wt_list) 
 
    f1, ax = plt.subplots(1,2,figsize = (14,5))
    # plot layout highlighting the transect
    iwt = [x in wt_list for x in turbines['VDC ID']]
    ax[0].scatter(turbines['X coordinate'],turbines['Y coordinate'],c='silver', s=6)
    ax[0].scatter(turbines['X coordinate'][iwt],turbines['Y coordinate'][iwt],c='black',s=6)
    ax[0].text(coords[0][0],coords[0][1],wt_list[0],{'ha': 'right'})
    ax[0].text(coords[-1][0],coords[-1][1],wt_list[-1],{'ha': 'right'})
    ax[0].axis('scaled')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['bottom'].set_visible(False)
    ax[0].spines['left'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].get_xaxis().set_ticks([])
    ax[0].get_yaxis().set_ticks([])

    #plot profiles of array efficiency ratio and mesoscale power ratio 
    for index, row in data.iterrows():
        eta = row.reindex(wt_list)
        ax[1].plot(dists,eta/eta[0])
    ax[1].plot(dists,ref_data/ref_data[0], marker='s', markerfacecolor='silver', markeredgecolor= 'k', color = 'k', linewidth = 3)
    #ax[1].legend(np.append(sim_name, 'Ref'),bbox_to_anchor=(1.15, 1))
    ax[1].set_ylim([0.4,1.2])    
    ax[1].set_ylabel('$\eta/\eta_{0}$')
    ax[1].set_title('Array efficiency ratio along transect '+wt_list[0]+'-'+wt_list[-1]+' ('+WDbin+', '+zLbin+')')
    ax[1].grid(True)
    for wt in range(n_wt):
        ax[1].text(dists[wt],0.5,wt_list[wt],rotation=90.,color='k')

    meso_P_ratio = meso_data.reindex(wt_list)
    meso_P_ratio = meso_P_ratio/meso_P_ratio[0]
    bx = ax[1].twinx()
    bx.plot(dists,meso_P_ratio,'--b')
    bx.set_ylabel('$(P_{0}/P)_{meso}$', color='b')

    ax[1].yaxis.set_major_locator(mtick.LinearLocator(9))
    bx.yaxis.set_major_locator(mtick.LinearLocator(9))

#    plt.errorbar(x, y, e, linestyle='None', marker='^')
#    x = dists
#    y = ref_data/ref_data[0]
#    e = ((1/ref_data[0]**2) * ref_data_std**2 + (ref_data**2/ref_data[0]**4) * ref_data_std[0]**2)**0.5 # asuuming independence     between ref_data and ref_data[0]
#    ax[1].errorbar(x,y,e, marker='^', linestyle='None')

    plt.tight_layout()
    plt.show()
  
    return ax,bx


def restrict_to_ts(data, ts):
    return data[~data.index.duplicated()].reindex(ts).dropna()

def flags_to_ts(scada_flags, min_data_availability):
    filter_condition = (100 - min_data_availability)/100 * scada_flags.shape[1]
    flags_filter = (scada_flags.sum(axis=1) < filter_condition) 
    return flags_filter[flags_filter].keys()

class BinAvrg:
    """
    Bin averaging class for analysis and plotting
    """
    
    def __init__(self,site_id,datefrom, dateto, wd_bins, wd_bins_label, zL_bins, zL_bins_label, wt_label,sim_id):
        self.site_id = site_id
        self.datefrom = datefrom
        self.dateto = dateto
        self.wd_bins = wd_bins
        self.wd_bins_label = wd_bins_label
        self.zL_bins = zL_bins
        self.zL_bins_label = zL_bins_label
        self.wt_label = wt_label
        self.sim_id = sim_id
        self.bin_size = {'wt':len(wt_label), 'wd':len(wd_bins_label),'zL':len(zL_bins_label),'sim':len(sim_id)}
        self.labels =   {'wt':wt_label     , 'wd':wd_bins_label     ,'zL':zL_bins_label,     'sim':sim_id}
    
    def __create_bin_map(self, time, wd, zL):
        wd[wd>self.wd_bins[-1]] = wd[wd>self.wd_bins[-1]]-360
        
        wd_bins_map = pd.cut(wd, self.wd_bins, labels = False)
        wd_bins_map = pd.Series( wd_bins_map, time)
        
        zL_bins_map = pd.cut(zL, self.zL_bins, labels = False)
        zL_bins_map = pd.Series( zL_bins_map, time)
        
        # join the two maps
        return pd.concat([wd_bins_map, zL_bins_map] , axis=1, keys=['wd', 'zL0'])
    
    def filter_s(self, mast, zref, scada_ts, s_bins):
        # get time
        time = mast.data['t'][self.datefrom:self.dateto].values.flatten()
        
        #get speed
        if not zref:  # point 
            s = mast.data['S'].values.flatten()
        else:         # profile 
            s = (mast.z_interp_data('S', self.datefrom, self.dateto, zref)).values.flatten()
            
        s = pd.Series( s, time)
        
        # filter out speed restrictions
        s = s[(s >= s_bins[0]) & (s <= s_bins[1])] #time dimension has to fit

        # filter out scada availability flags
        s = s.reindex(scada_ts)

        # clean up
        s = s.dropna()
        return s.index.values
    
    def read_ba_file(self, file_name):
        input_file = pd.read_csv(file_name)
        bin_averages = np.empty((len(input_file), len(self.wd_bins)-1, len(self.zL_bins_label)))
        for i_label in range(len(self.zL_bins_label)):
            for i_wd in range(len(self.wd_bins)-1):
                col_name = 'P' + str(i_wd+1) + self.zL_bins_label[i_label]
                for i_wt in range(len(input_file)):
                    bin_averages[i_wt,i_wd,i_label] = input_file[col_name][i_wt]
        return bin_averages
        
    def create_ts_to_bin_map(self, mast, zref, scada_ts = None):
        # get time
        time = mast.data['t'][self.datefrom:self.dateto].values.flatten()
        
        # create bins map
        if not zref:
            wd = mast.data['WD'][self.datefrom:self.dateto].values.flatten()
            zL = mast.data['zL0'][self.datefrom:self.dateto].values.flatten()
        else:
            wd = (mast.z_interp_data('WD', self.datefrom, self.dateto, zref)).values.flatten()
            zL = (mast.data['zL0'][self.datefrom:self.dateto]).values.flatten()

        map_array = self.__create_bin_map(time, wd, zL)


        # filter out scada availability flags
        if scada_ts is not None:
            map_array = restrict_to_ts (map_array,scada_ts)

        # clean up
        map_array = map_array.dropna()
        map_array = map_array.astype(int)
        
        # make the final 2d array of timestamps list
        timestamps_map = np.empty((len(self.wd_bins)-1, len(self.zL_bins)-1), dtype = object)
        for ix,iy in np.ndindex(timestamps_map.shape): # init it with empty lists
            timestamps_map[ix,iy] = []

        for index, row in map_array.iterrows(): # fill the lists with timestamps
            timestamps_map[row['wd']][row['zL0']].append(index)
            
        return timestamps_map
    
    def array_init(self,dims):
        # only accept 2 or 3 dim xarrays with 'wt', 'wd','zL','sim' dims
        if len(dims)==1:
            shape=(self.bin_size[dims[0]])
        elif len(dims)==2:
            shape=(self.bin_size[dims[0]], self.bin_size[dims[1]])
        elif len(dims)==3:
            shape=(self.bin_size[dims[0]], self.bin_size[dims[1]],self.bin_size[dims[2]])
        else:
            shape=(self.bin_size[dims[0]], self.bin_size[dims[1]],self.bin_size[dims[2]],self.bin_size[dims[3]])
            
        labels = {k:self.labels[k] for k in self.labels if k in dims}
        data = np.empty(shape)
        data.fill(np.nan)
        data = xr.DataArray(data, coords=labels, dims=dims)
        
        #for simpler arrays it is better to use pandas
        if len(dims)<2:
            return data.to_pandas()
        
        return data
    
    def compute_mean(self, ts, ts_bin_map):
        #clean up duplicates 
        ts = ts[~ts.index.duplicated()]
        #create output arrays
        (n_wd_bins, n_stab_bins) = ts_bin_map.shape
        (_ , n_wt) = ts.shape
              
        if len(ts.squeeze().shape) == 2: # wt
            mean = self.array_init(('wt', 'wd','zL'))
            std = xr.DataArray(np.empty((n_wt,n_wd_bins, n_stab_bins)), 
                            coords={'wt':self.wt_label, 'wd': self.wd_bins_label, 'zL':self.zL_bins_label}, 
                            dims=('wt', 'wd','zL'))
            # compute std and mean
            for i_wd in range(n_wd_bins):
                for i_stab in range(n_stab_bins):
                    ts_bin = ts.reindex(ts_bin_map[i_wd,i_stab])
                    mean[:,i_wd,i_stab] = ts_bin.mean()
                    std[:,i_wd,i_stab] =  ts_bin.std()
        else:
            mean = self.array_init(('wd','zL'))
            std = xr.DataArray(np.empty((n_wd_bins, n_stab_bins)), 
                            coords={'wd': self.wd_bins_label, 'zL':self.zL_bins_label}, 
                            dims=('wd','zL'))
            # compute std and mean
            for i_wd in range(n_wd_bins):
                for i_stab in range(n_stab_bins):
                    ts_bin = ts.reindex(ts_bin_map[i_wd,i_stab])
                    mean[i_wd,i_stab] = ts_bin.mean()[0]
                    std[i_wd,i_stab] =  ts_bin.std()[0]
        
        return mean, std
    def plot_heatmaps(self,data,sub_plt_size = (1.7,4),n_plot_cols = 4, figcaption = '', title=''):
        #remove empty simulations
        #mask = [len(elem) != 0 for elem in data] 
        #data = [data[i] for i in range(len(mask)) if mask[i]]
        #sim_id = [sim_id[i] for i in range(len(mask)) if mask[i]]
        sim_id = data.coords['sim'].values.flatten()
        
        
        n_sim = len(data)

        #Xlabel = '$WD_{ref}$'
        #Ylabel =  '$z/L_0$'

        max_data = np.nanmax(np.absolute(data))
        z_levels = np.linspace(-max_data,max_data,14)
        cmap = plt.get_cmap('bwr')

        if n_sim < n_plot_cols:
            n_plot_cols = n_sim

        n_plot_rows = int(n_sim / n_plot_cols)
        if n_plot_rows < (n_sim / n_plot_cols): 
            n_plot_rows = n_plot_rows+1

        figname = self.site_id+'_heatmaps.png'
        fig, ax = plt.subplots(n_plot_rows, n_plot_cols, sharex='col', sharey='row', 
                               figsize=(n_plot_cols*sub_plt_size[0] ,n_plot_rows*sub_plt_size[1]))

        for iax in range (0,n_sim):
            if n_plot_rows>1:
                index = np.unravel_index(iax,(n_plot_rows,n_plot_cols))
            else: 
                index = iax
            
            #colour bar for the last plot in row
            if iax%n_plot_cols == (n_plot_cols-1):
                cbar = True
            else:
                cbar = False

            sns.heatmap(data[iax].to_pandas(), ax = ax[index], cmap=cmap, 
                    vmin=z_levels.min(), vmax=z_levels.max(),
                    cbar_kws={'boundaries':z_levels},
                    cbar=cbar,
                    xticklabels = True, yticklabels=True,
                    linewidths=.1)
            ax[index].set_facecolor('grey')
            ax[index].set_title(sim_id[iax]+title)

        plt.tight_layout()
        #plt.savefig(figname, dpi=300, bbox_inches='tight')

        plt.show()

        display(Markdown(figcaption))
