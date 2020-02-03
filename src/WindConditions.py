import sys
import os
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

def time_stamp(year,month,day,hour,minute,second):
    """ Abstraction method for creating compatible time stamps"""
    return datetime.datetime(year,month,day,hour,minute,second)   

def timedelta_to_resample_string(timedelta):
    """
    Converts timedelta to a string rappresentation
    compatible with pandas resample
    """
    text = ''
    if timedelta.days != 0:
        text += '%dD' % timedelta.days
    if timedelta.seconds > 0:
        text += '%ds' % timedelta.seconds

    return text

class WindConditions:
    """
    Wind condition class for plotting and analysis of met mast data
    """
    
    # Constants
    G = 9.81    # [m s-2]
    P0 = 100000 # Reference pressure [Pa]
    T0 = 300    # Reference temperature for perturbation temperature [K]
    KAPPA = 0.2854  # Poisson constant (R/Cp)
    R_AIR = 287.058  # Specific gas constant for dry air [J kg-1 K-1]
    CP_AIR = 1005   # Specific heat of air [J kg-1 K-1]
    OMEGA = 7.2921159e-5    # angular speed of the Earth [rad/s]
    K = 0.41    # von Karman constant
    TIME_UNITS = "seconds since 1970-01-01 00:00:00.00 UTC"


    def __init__(self,filetend, siteID = None, datefrom = None , dateto = None, filetype = 'profile'):
        self.siteID = siteID
        #self.latitude = latitude      # degrees N 
        #self.longitude = longitude      # degrees E
        #self.coriolis_parameter  = 2*self.OMEGA*np.sin(self.latitude*np.pi/180) 
        self.data = {}
        
        
        self.__read_inputs(filetend, datefrom, dateto, filetype)
        
        
    def __read_inputs(self,file, datefrom, dateto, filetype):
        if not(os.path.isfile(file)):
            print(file + 'does not exist')
        else:
            f = netCDF4.Dataset(file, 'r')
            times = f.variables['Times'][:]
            times = netCDF4.num2date(times,self.TIME_UNITS)
            mask = np.logical_and(times >= datefrom, times < dateto)
            idates = np.where(mask)[0]
            times= times[mask]
            sampling_time = timedelta_to_resample_string(times[1] - times[0])
            
            if filetype == 'profile':
                z = f.variables['height'][:]
                self.data['z'] = z
                self.data['t'] = pd.DataFrame(f.variables['Times'][idates], index = times).resample(sampling_time).mean()       
                self.data['U'] = pd.DataFrame(f.variables['U'][idates,:], index = times, columns = z)#.resample(sampling_time).mean() 
                self.data['V'] = pd.DataFrame(f.variables['V'][idates,:], index = times, columns = z)#.resample(sampling_time).mean() 
                self.data['Th'] = pd.DataFrame(f.variables['POT'][idates,:], index = times, columns = z)#.resample(sampling_time).mean() 
                self.data['us'] = pd.DataFrame(f.variables['UST'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['wt'] = pd.DataFrame(f.variables['wt'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['T2'] = pd.DataFrame(f.variables['T2'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['TSK'] = pd.DataFrame(f.variables['TSK'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['Psfc'] = pd.DataFrame(f.variables['PSFC'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['Ug'] = pd.DataFrame(f.variables['Ug'][idates], index = times, columns = z)#.resample(sampling_time).mean() 
                self.data['Vg'] = pd.DataFrame(f.variables['Vg'][idates], index = times, columns = z)#.resample(sampling_time).mean() 
                self.data['Uadv'] = pd.DataFrame(f.variables['UADV'][idates], index = times, columns = z)#.resample(sampling_time).mean() 
                self.data['Vadv'] = pd.DataFrame(f.variables['VADV'][idates], index = times, columns = z)#.resample(sampling_time).mean() 
                self.data['Thadv'] = pd.DataFrame(f.variables['POT_ADV'][idates], index = times, columns = z)#.resample(sampling_time).mean() 
                f.close()
                #Nz = len(z)
                self.data['S'] = (self.data['U']**2 + self.data['V']**2)**0.5
                self.data['WD'] = 180. + np.arctan2(self.data['U'],self.data['V'])*180./np.pi

                # Replace with 10*RMOL (RMOL directly from WRF)
                self.data['zL0'] = 10./(- self.data['us']**3/(self.K*(self.G/self.data['T2'])*self.data['wt'])) # Stability parameter, reference to classify wind conditions 
            elif filetype == 'point':
                self.data['t'] = pd.DataFrame(f.variables['Times'][idates], index = times)#.resample(sampling_time).mean()       
                self.data['U'] = pd.DataFrame(f.variables['U'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['V'] = pd.DataFrame(f.variables['V'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['TKE'] = pd.DataFrame(f.variables['TKE'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['us'] = pd.DataFrame(f.variables['UST'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['HFX'] = pd.DataFrame(f.variables['HFX'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['T2'] = pd.DataFrame(f.variables['T2'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['TSK'] = pd.DataFrame(f.variables['TSK'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['POT'] = pd.DataFrame(f.variables['POT'][idates], index = times)#.resample(sampling_time).mean() 
                self.data['zL0'] = 10./pd.DataFrame(f.variables['L'][idates], index = times)#.resample(sampling_time).mean() 
                f.close()
                self.data['S'] = (self.data['U']**2 + self.data['V']**2)**0.5
                self.data['WD'] = 180. + np.arctan2(self.data['U'],self.data['V'])*180./np.pi
            else:
                print('filetype not valid')
            

    def reduce_to_ts(self,time_series, filetype):
        """ Reduce all the data to the provided time stamps"""
        ts_set = set(time_series)
        new_time = self.data['t'][:][self.data['t'][0].apply(lambda x: x in ts_set)].index.values
        
        if filetype == 'profile':
            self.data['t'] = self.data['t'].reindex(new_time)
            self.data['U'] =  self.data['U'].reindex(new_time)
            self.data['V'] = self.data['V'].reindex(new_time)
            self.data['Th'] = self.data['Th'].reindex(new_time)
            self.data['us'] = self.data['us'].reindex(new_time)
            self.data['wt'] = self.data['wt'].reindex(new_time)
            self.data['T2'] = self.data['T2'].reindex(new_time)
            self.data['TSK'] = self.data['TSK'].reindex(new_time)
            self.data['Psfc'] = self.data['Psfc'].reindex(new_time)
            self.data['Ug'] = self.data['Ug'].reindex(new_time)
            self.data['Vg'] = self.data['Vg'].reindex(new_time)
            self.data['Uadv'] = self.data['Uadv'].reindex(new_time)
            self.data['Vadv'] = self.data['Vadv'].reindex(new_time)
            self.data['Thadv'] = self.data['Thadv'].reindex(new_time)
            self.data['S'] = self.data['S'].reindex(new_time)
            self.data['WD'] = self.data['WD'].reindex(new_time)
            self.data['zL0'] = self.data['zL0'].reindex(new_time)
        elif filetype == 'point':
            self.data['t'] = self.data['t'].reindex(new_time)
            self.data['U'] =  self.data['U'].reindex(new_time)
            self.data['V'] = self.data['V'].reindex(new_time)
            self.data['us'] = self.data['us'].reindex(new_time)
            self.data['TKE'] = self.data['TKE'].reindex(new_time)
            self.data['HFX'] = self.data['HFX'].reindex(new_time)
            self.data['T2'] = self.data['T2'].reindex(new_time)
            self.data['TSK'] = self.data['TSK'].reindex(new_time)
            self.data['POT'] = self.data['POT'].reindex(new_time)
            self.data['S'] = self.data['S'].reindex(new_time)
            self.data['WD'] = self.data['WD'].reindex(new_time)
            self.data['zL0'] = self.data['zL0'].reindex(new_time)
        else:
            print('filetype not valid')
                
    
    def z_interp_data(self,var, datefrom, dateto, zref):
        if var == 'WD':
            res = interp1d(self.data['z'], self.data[var][self.data[var].columns.values][datefrom:dateto],kind = 'nearest')(zref)
        else:
            res = interp1d(self.data['z'], self.data[var][self.data[var].columns.values][datefrom:dateto])(zref)
        return pd.DataFrame(res, index = self.data['t'][datefrom:dateto].index)
        
    def plot_timeseries(self,datefromplot,datetoplot,zref):
        
        S = self.z_interp_data('S', datefromplot, datetoplot, zref)
        WD = self.z_interp_data('WD', datefromplot, datetoplot, zref)
        Th = self.z_interp_data('Th', datefromplot, datetoplot, zref)
        #us = self.data['us'][datefromplot:datetoplot]
        zL0 = self.data['zL0'][datefromplot:datetoplot]
        
        Splot = pd.concat([S], axis = 1, keys = ['WRF'])
        WDplot = pd.concat([WD], axis = 1, keys = ['WRF'])
        Thplot = pd.concat([Th], axis = 1, keys = ['WRF']) 
        #usplot = pd.concat([us], axis = 1, keys = ['WRF'])
        zL0plot = pd.concat([zL0], axis = 1, keys = ['WRF']) 
        # Consider adding wind farm equivalent wind speed and direction from SCADA to compare against
        
        fig,axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6),sharex=True, sharey=False)
        fig.suptitle('z = ' + str(zref) + 'm')
        Splot.plot(ax=axes[0,0], style=['-','.'], color =['k','silver'],legend=False, grid=True); axes[0,0].set_title('$S$ [$m s^{-1}$]')
        WDplot.plot(ax=axes[0,1], style=['-','.'], color =['k','silver'], grid=True); axes[0,1].set_title('$WD$ ['+u'\N{DEGREE SIGN}'+']')
        #usplot.plot(ax=axes[1,0], style=['.','-'], color =['silver','k'],legend=False, grid=True); axes[1,0].set_title('$u_*$ [$m s^{-1}$]')
        zL0plot.plot(ax=axes[1,0], style=['-','.'], color =['k','silver'],legend=False, grid=True); axes[1,0].set_title('$z/L$')
        Thplot.plot(ax=axes[1,1], style=['-','.'], color =['k','silver'],legend=False, grid=True); axes[1,1].set_title('$\Theta$ ['+u'\N{DEGREE SIGN}'+'C]')
        #figname = siteID+'_zref_timeseries.png'
        #plt.savefig(figname, bbox_inches='tight', dpi=300)


    def plot_ABL(self,datefromplot,datetoplot,z_lim,z_mark1 = 0,z_mark2 = 0):
        taxis = self.data['t'][0][datefromplot:datetoplot].values
        zaxis = self.data['z'] 
        [X,Y] = np.meshgrid(taxis,zaxis)

        Z = [self.data['U'][datefromplot:datetoplot].values.T,
             self.data['Uadv'][datefromplot:datetoplot].values.T,
             -self.data['Vg'] [datefromplot:datetoplot].values.T,
             self.data['V'] [datefromplot:datetoplot].values.T,
             self.data['Vadv'] [datefromplot:datetoplot].values.T,
             self.data['Ug'][datefromplot:datetoplot].values.T]
        Zvarname = ['$U$','$U_{adv}$','$U_{pg}$',
          '$V$','$V_{adv}$','$V_{pg}$']

        Zlevels = np.linspace(-24,24,13, endpoint=True)
        
        taxis_label = 'Period: ' + datefromplot.strftime('%Y-%m-%d %H:%M') + ' to '+ datetoplot.strftime('%Y-%m-%d %H:%M')         
        zaxis_label = '$z$ [$m$]'
        # hoursFmt = mdates.DateFormatter('%d')
        
        xrotor = np.array([netCDF4.date2num(datefromplot,self.TIME_UNITS), netCDF4.date2num(datetoplot,self.TIME_UNITS)])
        
        daysPlot=[(datefromplot + datetime.timedelta(i)).strftime("%d") 
                  for  i in range((datetoplot - datefromplot).days + 1)]
        
        idaysPlot=[netCDF4.date2num(datefromplot + datetime.timedelta(i),self.TIME_UNITS) 
                  for  i in range((datetoplot - datefromplot).days + 1)]
        
        fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(8,5))
        Zcmap = plt.get_cmap('bwr') #bwr
        
        for iax in range (0,6):
            
            ix,iy = np.unravel_index(iax,(2,3))
            CF = ax[ix,iy].contourf(X,Y,Z[iax], Zlevels, cmap=Zcmap)
            ax[ix,iy].plot(xrotor,z_mark1,'--k')
            ax[ix,iy].plot(xrotor,z_mark2,'--k')
            ax[ix,iy].set_ylim([10, z_lim])
            ax[ix,iy].set_yscale('log')
            ax[ix,iy].set_title(Zvarname[iax])
            
            if iax >= 3:
                plt.sca(ax[ix,iy])
                plt.xticks(idaysPlot,daysPlot, color='black')
            
            
        ax[0,0].set_ylabel(zaxis_label); ax[1,0].set_ylabel(zaxis_label)
        ax[1,1].set_xlabel(taxis_label)
        
        
        fig.subplots_adjust(right=0.92)
        cbar_ax = fig.add_axes([0.95, 0.12, 0.025, 0.75])
        cbar = fig.colorbar(CF, cax=cbar_ax)    
        cbar.ax.set_xlabel('[$m s^{-1}$]',labelpad = -290, x = 1.3)


    def plot_windrose(self, zref, datefromplot = None, datetoplot = None):
        # Plot windorse
        #figname = siteID + '_windrose.png'
        
        if datefromplot is None:
            datefromplot = self.data['t'].index[0]
        if datetoplot is None:
            datetoplot = self.data['t'].index[-1]

        #fetch all the data
        S = self.z_interp_data('S', datefromplot, datetoplot, zref)[0]
        WD = self.z_interp_data('WD', datefromplot, datetoplot, zref)[0]

        rose = pd.concat([S ,WD], axis = 1, keys = ['speed','direction']).dropna()

        ax = WindroseAxes.from_ax()
        ax.bar(rose.direction, rose.speed, normed=True, bins=np.arange(0.,25.,5.) , opening=0.8, edgecolor='black')
        ax.set_legend()

    def analyse_stability(self,WDbins,WDbins_label,zLbins,zLbins_label, zref,min_S = -1, max_S = 999, datefromplot = None, datetoplot = None):
        if datefromplot is None:
            datefromplot = self.data['t'].index[0]
        if datetoplot is None:
            datetoplot = self.data['t'].index[-1]

        # Plot stability per wind direction sector
        Sbins = np.hstack((np.arange(0,24,1)))     # velocity bins
        Sbins_label = Sbins[0:-1]+1

        #fetch all the data
        if not zref:
            S = self.data['S'][datefromplot:datetoplot]
            WD = self.data['WD'][datefromplot:datetoplot]
        else:
            S = self.z_interp_data('S', datefromplot, datetoplot, zref)[0]
            WD = self.z_interp_data('WD', datefromplot, datetoplot, zref)[0]
        
        zL0 = self.data['zL0'][datefromplot:datetoplot]
        
        clean_data = pd.concat([S ,WD,zL0], axis = 1, keys = ['S','WD','zL0']).dropna()
        s_in_range =  np.logical_and(clean_data['S'][0] >= min_S , clean_data['S'][0] <= max_S)
        
        #clean_data = clean_data[:][:][s_in_range]

        S = clean_data['S'][0][s_in_range]
        WD = clean_data['WD'][0][s_in_range]
        zL0 = clean_data['zL0'][0][s_in_range]

        #process the data
        bins = [Sbins,zLbins]
        x = S.values.flatten()
        y = zL0.values.flatten()

        values = S.values.flatten()
        statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='count', bins=bins, expand_binnumbers = True)
        N_SzL = pd.DataFrame(statistic, index=Sbins_label, columns=zLbins_label)

        bins = [WDbins,zLbins]
        x = WD.values
        x[x>WDbins[-1]] = x[x>WDbins[-1]]-360
        y = zL0.values.flatten()

        values = S.values.flatten()
        statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='count', bins=bins, expand_binnumbers = True)
        N_WDzL = pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label)
        N_zL = np.sum(N_WDzL, axis = 0).rename('pdf')
        N_WD = np.sum(N_WDzL, axis = 1).rename('pdf')
        statistic, xedges, yedges, binnumber = stats.binned_statistic_2d(x, y, values, statistic='mean', bins=bins, expand_binnumbers = True)
        S_WDzL = pd.DataFrame(statistic, index=WDbins_label, columns=zLbins_label)

        

        return N_WDzL, N_SzL, N_zL, N_WD, S_WDzL

    def plot_stability(self,WDbins,WDbins_label,zLbins,zLbins_label,zref,min_S = -1, max_S = 999,datefromplot = None, datetoplot = None):
        N_WDzL,N_SzL,N_zL,N_WD,S_WDzL = self.analyse_stability(
                                        WDbins,WDbins_label,zLbins,zLbins_label,zref,min_S,max_S,datefromplot, datetoplot)
        
        N_S = np.sum(N_SzL, axis = 1).rename('pdf')
        Nnorm_WDzL = N_WDzL.div(N_WD, axis=0)
        Nnorm_SzL = N_SzL.div(N_S, axis=0)

        Sbins = np.hstack((np.arange(0,24,1)))     # velocity bins
        Sbins_label = Sbins[0:-1]+1

        # plot
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        cmap = plt.get_cmap('bwr')
        NzL = len(zLbins_label)
        zLcolors = np.flipud(cmap(np.linspace(0.,NzL,NzL)/NzL))
        ax1=Nnorm_WDzL.plot.bar(ax=axes[0], stacked=True, color=zLcolors, align='center', width=1.0, legend=False, rot=90, use_index = False)
        ax2=(N_WD/N_WD.sum()).plot(ax=axes[0], secondary_y=True, style='k',legend=False, rot=90, use_index = False)
        ax2.set_xticklabels(WDbins_label)
        ax1.set_title('WRF: Wind direction vs stability')
        ax1.set_ylabel('$pdf_{norm}$($z/L_0$)')
        ax2.set_ylabel('$pdf$($WD_{ref}$), $z_{ref}$ = '+str(zref)+' m', rotation=-90, labelpad=15)
        ax1.set_yticks(np.linspace(0,1.,6))
        ax1.set_ylim([0,1.])
        ax2.set_yticks(np.linspace(0,0.2,6))

        ax3=Nnorm_SzL.plot.bar(ax=axes[1], stacked=True, color=zLcolors, align='center', width=1.0,legend=False, rot=90, use_index = False)
        ax4=(N_S/N_S.sum()).plot(ax=axes[1], secondary_y=True, style='k',legend=False, rot=90, use_index = False)
        ax4.set_xticklabels(Sbins_label)
        ax3.set_title('WRF: Wind speed vs stability')
        ax3.set_ylabel('$pdf_{norm}$($z/L_0$)')
        ax4.set_ylabel('$pdf$($S_{ref}$), $z_{ref}$ = '+str(zref)+' m', rotation=-90, labelpad=15)
        ax3.set_yticks(np.linspace(0,1.,6))
        ax3.set_ylim([0,1.])
        ax4.set_yticks(np.linspace(0,0.2,6))
        h1, l1 = ax3.get_legend_handles_labels()
        h2, l2 = ax4.get_legend_handles_labels()
        plt.legend(h1+h2, l1+l2, bbox_to_anchor=(1.4, 1))
        plt.tight_layout(pad=0.4, w_pad=1.2, h_pad=1.2)    

        #figname = siteID + '_zLvsWD_dist.png'
        #plt.savefig(figname, dpi=300, bbox_inches='tight')
        
        return N_WDzL, N_SzL, N_zL, N_WD, S_WDzL
