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

def plot_transect(data,val_data,meso_data,wt_list,turbines,rot_d,sim_name,WDbin,zLbin,highlight,ylim1,ylim2,figsize,wtref):
    n_wt = len(wt_list)
    n_sim = data.shape[0]
    
    #compute distances to first turbine
    a = turbines.loc[turbines['VDC ID'] == wt_list[0],['X coordinate','Y coordinate']].values.flatten()            
    dists = []
    coords = np.zeros((n_wt,2))
    for wt in range(n_wt):
        b = turbines.loc[turbines['VDC ID'] == wt_list[wt],['X coordinate','Y coordinate']].values.flatten()
        dists.append(((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5 / rot_d)
        coords[wt,:]=b
         
    f1, ax = plt.subplots(1,2,figsize = figsize)
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
       
    if wtref == 'ref':
        data[wt_list].div(data.mean(axis = 1), axis=0).T.plot(ax = ax[1], color = 'lightgrey')
        data[wt_list].div(data.mean(axis = 1), axis=0).T.plot(y = highlight, ax = ax[1], label = sim_name)
        data[wt_list].div(data.mean(axis = 1), axis=0).T.plot(y = 'ensemble', ax = ax[1], linewidth = 3, 
                                                              color = 'k',linestyle='-.', label = 'ensemble') 
        if np.sum(data.loc['ensemble']-val_data)!=0.0:
            val_data[wt_list].div(val_data.mean()).plot(marker='s', markerfacecolor='cyan', linewidth = 3, 
                                                        markeredgecolor= 'k', markersize = 8, color = 'cyan', linestyle='--', 
                                                        label = 'Observations') 
    else:
        data[wt_list].div(data[wtref], axis=0).T.plot(ax = ax[1], color = 'lightgrey')        
        data[wt_list].div(data[wtref], axis=0).T.plot(y = highlight, ax = ax[1], label = sim_name)
        data[wt_list].div(data[wtref], axis=0).T.plot(y = 'ensemble', ax = ax[1], linewidth = 3, 
                                                      color = 'k', linestyle='-.', label = 'ensemble')
        if np.sum(data.loc['ensemble']-val_data)!=0.0:
            val_data[wt_list].div(val_data[wt_ref]).plot(marker='s', markerfacecolor='cyan', linewidth = 3,
                       markeredgecolor= 'k',  markersize = 8, color = 'cyan', linestyle='--', label = 'Observations',
                       yerr = val_data[wt_list].div(val_data[wt_ref]))
    
    current_handles, current_labels = plt.gca().get_legend_handles_labels() 
    ax[1].legend(current_handles[n_sim:], current_labels[n_sim:],bbox_to_anchor=(1.6, 1)) # avoid lightgrey sims in the legend
    ax[1].set_ylim(ylim1)
    if wtref == 'ref':
        ax[1].set_ylabel('Net Power Ratio: $P/P_{ref}$')  
    else:
        ax[1].set_ylabel(f'Net Power Ratio: $P/P_{wtref}$')  
    #ax[1].set_xlabel('Distance from first turbine (D)')
    ax[1].set_title('Transect '+wt_list[0]+'-'+wt_list[-1]+' ('+WDbin+', '+zLbin+')')
    ax[1].grid(True)
#    for wt in range(n_wt):
#        ax[1].text(dists[wt],0.5,wt_list[wt],rotation=90.,color='k')
    
    meso_P_ratio = meso_data.reindex(wt_list)
    if wtref == 'ref':
        meso_P0 = meso_data.mean()
    else:
        meso_P0 = meso_data.loc[wtref]
    meso_P_ratio = meso_P_ratio/meso_P0
    bx = ax[1].twinx()
    bx.plot(wt_list,meso_P_ratio,'--b')
    if wtref == 'ref':
        bx.set_ylabel('Mesoscale Gross Power Ratio: $P(S)/P(S_{ref})$', color='b')
    else:
        bx.set_ylabel(f'Mesoscale Gross Power Ratio: $P(S)/P(S_{wtref})$', color='b')   

    bx.set_ylim(ylim2)
    ax[1].yaxis.set_major_locator(mtick.LinearLocator(9))
    bx.yaxis.set_major_locator(mtick.LinearLocator(9))

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
