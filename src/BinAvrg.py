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

def plot_wf_layout(x,y,labels = [], figsize=(8,8),data = [],vmin = np.nan,vmax = np.nan):
    fig= plt.figure(figsize=figsize)
    
    #1D data - make a single plot
    #if len(data.shape) == 1:
    #main plot
    if len(data)==0:
        plt.scatter(x,y)
    else:
        if np.isnan(vmax):
            vmax = np.nanmax(data)
        if np.isnan(vmin):
            vmin = np.nanmin(data)
        
        # blue white red plot centered on white
        if vmin<0:
            vmax = max(vmax,-vmin)
            vmin = -vmax
            cmap=plt.cm.get_cmap('bwr', 15)
            sc = plt.scatter(x,y,marker='o',c=data,cmap=cmap,edgecolors ='k')
        else:# jet plot
            cmap=plt.cm.get_cmap('jet', 14)
            sc = plt.scatter(x,y,c=data,cmap=cmap)
        
        plt.colorbar(sc)
        plt.clim(vmin, vmax);

    #labeled wind turbines    
    for i in range(len(labels)):
        plt.text(x[i]+0.4, y[i]+0.4, labels[i][3:], fontsize=9) #the labels prefix is removed
    #else: #2D data - make multiplot
    # TODO
    plt.axis('scaled')
    plt.show()

def plot_transect(data,ref_data,ref_data_max,wt_list,turbines,rot_d,figsize=(8,8)):
    #compute distances
    a = turbines.loc[turbines['VDC ID'] == wt_list[0],['X coordinate','X coordinate']].values.flatten()
    dists = []
    for wt in wt_list:
        b = turbines.loc[turbines['VDC ID'] == wt,['X coordinate','X coordinate']].values.flatten()
        dists.append(((a[0]-b[0])**2+(a[1]-b[1])**2)**0.5 / rot_d)
    
    
    ref_data = ref_data.reindex(wt_list)
    ref_data_std = ref_data_max.reindex(wt_list) - ref_data
    ref_data_std[0] = 0
    f1, ax = plt.subplots(1,2,figsize = (10,4))

    ax[0].scatter(turbines['X coordinate'],turbines['Y coordinate'],c=[x in wt_list for x in turbines['VDC ID']])
    ax[0].axis('scaled')

    for index, row in data.iterrows():
        eta = row.reindex(wt_list)
        ax[1].plot(dists,eta/eta[0])

    #plt.errorbar(x, y, e, linestyle='None', marker='^')
    ax[1].errorbar(dists,ref_data/ref_data[0],ref_data_std/ref_data[0], marker='^', linestyle='None')

    legend = np.append(data.index.values, 'Ref')
    _ = plt.legend(legend,loc='upper right')

    plt.tight_layout()
    plt.show()

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
