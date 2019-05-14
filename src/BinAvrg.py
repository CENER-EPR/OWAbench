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
    
    def __init__(self,datefrom, dateto, wd_bins, wd_bins_label, zL_bins, zL_bins_label):
        self.datefrom = datefrom
        self.dateto = dateto
        self.wd_bins = wd_bins
        self.wd_bins_label = wd_bins_label
        self.zL_bins = zL_bins
        self.zL_bins_label = zL_bins_label
    
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
    
    
    def compute_mean(self, ts, ts_bin_map):
        #create output arrays
        (n_wd_bins, n_stab_bins) = ts_bin_map.shape
        (_ , n_wt) = ts.shape
        mean = np.empty((n_wt,n_wd_bins, n_stab_bins))
        std = np.empty((n_wt,n_wd_bins, n_stab_bins))
        
        # compute std and mean
        for i_wd in range(n_wd_bins):
            for i_stab in range(n_stab_bins):
                mean[:,i_wd,i_stab] = ts.loc[ts_bin_map[i_wd,i_stab]].mean()
                std[:,i_wd,i_stab] = ts.loc[ts_bin_map[i_wd,i_stab]].std()
        
        return mean, std