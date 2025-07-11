# Lasing-on Analysis Class

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from scipy.interpolate import interp1d
from xtcav_processor import XTCAVProcessor, ECHARGE

class XTCAVLasingOnReconstruction(XTCAVProcessor):

    def __init__(self, data_source, env=None, iteration='ok', fov=(120,240),
        dark_reference=None, lasing_off=None, n_pulses=1, thresh_current=0.1,
        verbose=True, _test_xy=False, _preindex=False):
        """
        Parameters
        ----------
        data_source : a psana DataSource instance
            Must point to a dark run
        env : None or a psana.Env
            if None uses dataSouce.env()
        iteration : string in ('all', 'ok', 'frames'):
            Determines which data is traversed when using the shot iterator.
            'all' iterates over all shots. 'ok' iterates over shots with all data
            present.  'frames' iterates over all data with XTCAV camera data present.
        fov : tuple
            The field-of-view to zoom in about each shot's center-of-mass, in pixels.
            This value is used both for display as well as for data filtering. For
            filtering, if more than 10% of the zoom-box along a single axis extends
            beyond the camera frame, the shot is not used.
        dark_reference : XTCAVDarkReference or 2D array or None
            Object containing analysis from the dark reference run or average dark
            reference array. If None is passed, this should be set later using
            .set_darkreference().
        lasing_off : XTCAVLasingOffReference or None
            Object containing analysis from the lasing off reference run. If None
            is passed, this should be set later using .set_lasing_off_reference().
        n_pulses : integer
            The number of pulses per shot
        thresh_current : number
            Disregard power reconstruction results wherever the electron current
            is below this fraction of the maximum
        verbose : bool
            Toggle verbosity
        _test_xy : bool
            Internal testing flag
        _preindex : bool
            Toggles indexing events on instantiation. If True, loops through all
            events and determines which data types (ebeam, gas detector, XTCAV
            camera frame) are present for each event.
        """
        XTCAVProcessor.__init__(
            self,
            data_source=data_source,
            env=env,
            iteration=iteration,
            fov=fov,
            dark_reference=dark_reference,
            n_pulses=n_pulses,
            verbose=verbose,
            _test_xy=_test_xy,
            _preindex=_preindex,
        )
        self.set_lasing_off_reference(lasing_off)
        self.set_current_threshold(thresh_current)
        self._recons = []
        pass

    def set_lasing_off_reference(self, lasing_off):
        """ Sets the lasing-off reference
        """
        self._lasing_off = lasing_off
    @property
    def lasing_off(self):
        return self._lasing_off
    @property
    def reference_profiles(self):
        try:
            return self.lasing_off.reference_profiles
        except AttributeError:
            return None

    def set_current_threshold(self,thresh_current):
        self._thresh_current = thresh_current
    @property
    def thresh_current(self):
        return self._thresh_current

    @property
    def recons(self):
        return self._recons
    @property
    def agreement_stats(self):
        return self._agreement_stats


    ### Power Profiles ###

    def get_single_shot_power_profile(self,idx):
        """
        Process a single shot, using the no lasing references to retrieve the
        power profile for each X-ray pulse.
        
        Parameters
        ----------
        idx : int
            The 'good' shot index

        Returns
        -------
        (dictionary) : the pulse info, in the following keys/values:

            Key                         Value Shape             Value
            ---                         -----------             -----
            't'                         (T,)                    Time axis (fs)
            'lasingenergyperpulse_ECOM' (N_pulses,)             Pulse X-ray energy using the center-of-mass approach (J)
            'lasingenergyperpulse_ERMS' (N_pulses,)             Pulse X-ray energy using the energy dispersion approach (J)
            'powerECOM'                 (N_pulses,T)            Power, reconstructed via ECOM (GW)
            'powerERMS'                 (N_pulses,T)            Power, reconstructed via ERMS (GW)
            'powerrawECOM'              (N_pulses,T)            Raw power (no gas detector normalization) reconstructed via ECOM (GW)
            'powerrawERMS'              (N_pulses,T)            Raw power (no gas detector normalization) reconstructed via ERMS (GW)
            'powerAgreement'            (N_pulses,)             Agreement factor in the power profiles between the two methods
            'lasingECurrent'            (N_pulses,T)            Electron current for the lasing trace vs. time (electrons/s)
            'nolasingECurrent'          (N_pulses,T)            Electron current for the non-lasing trace vs. time (electrons/s)
            'lasingECOM'                (N_pulses,T)            Lasing energy centers-of-mass vs. time (MeV)
            'nolasingECOM'              (N_pulses,T)            Non-lasing energy centers-of-mass vs. time (MeV)
            'lasingERMS'                (N_pulses,T)            Lasing energy dispersion vs. time (MeV)
            'nolasingERMS'              (N_pulses,T)            Non-lasing energy dispersion vs. time (MeV)
            'pulse_delay'               (N_pulses,)             Time delay of each pulse with respect to the first one (fs)
            'pulse_delaychange'         (N_pulses,)             Difference in time delay from the non-lasing reference (fs)
            'pulse_energydiff'          (N_pulses,)             Energy difference of each pulse with respect to the first one (MeV)
            'pulse_energydiffchange'    (N_pulses,)             Difference in energy differences from the non-lasing reference (MeV)
            'xrayenergy'                integer                 Total x-ray energy measured by the gas detector (J)
            'N_pulses'                  integer                 Number of pulses in the shot
            'groupnum'                  integer                 Group number indexing the lasing-off reference profile
        """
        # Get number of pulses
        n_pulses = self.n_pulses

        # Get shot statistics and metadata
        imageStats = []
        for i in range(n_pulses):
            imageStats.append(self._pulse_statistics[i][idx])
        shotToShot = self._shot_to_shot_params[idx]
        physicalUnits = self._physical_units[idx]

        # Get reference profiles
        reference_profiles = self.reference_profiles

        # Get info...
        t = reference_profiles['t']                         # time vector
        dt = (t[-1]-t[0])/(t.size-1)                        # time step
        Nelectrons = shotToShot['dumpecharge']/ECHARGE      # number of electrons in the shot
        
        # Containers
        pulse_delay = np.zeros(n_pulses, dtype=np.float64)               # Time delay of each pulse with respect to the first one (fs)
        pulse_delaychange = np.zeros(n_pulses, dtype=np.float64)         # Difference in time delay from the non-lasing reference (fs)
        pulse_energydiff = np.zeros(n_pulses, dtype=np.float64)          # Energy difference of each pulse with respect to the first one (MeV)
        pulse_energydiffchange = np.zeros(n_pulses, dtype=np.float64)    # Difference in energy differences from the non-lasing reference (MeV)
        pulse_lasingenergyCOM = np.zeros(n_pulses, dtype=np.float64)     # Pulse X-ray energy using the center of mass approach (J)
        pulse_lasingenergyRMS = np.zeros(n_pulses, dtype=np.float64)     # Pulse X-ray energy using the dispersion of mass approach (J)
        powerAgreement = np.zeros(n_pulses, dtype=np.float64)            # Agreement factor between the two methods
        lasingECurrent = np.zeros((n_pulses,t.size), dtype=np.float64)   # Electron current for the lasing trace vs. time (electrons/s)
        nolasingECurrent = np.zeros((n_pulses,t.size), dtype=np.float64) # Electron current for the non-lasing trace vs. time (electrons/s)
        lasingECOM = np.zeros((n_pulses,t.size), dtype=np.float64)       # Lasing energy centers-of-mass vs. time (MeV)
        nolasingECOM = np.zeros((n_pulses,t.size), dtype=np.float64)     # Non-lasing energy centers-of-mass vs. time (MeV)
        lasingERMS = np.zeros((n_pulses,t.size), dtype=np.float64)       # Lasing energy dispersion vs. time (MeV)
        nolasingERMS = np.zeros((n_pulses,t.size), dtype=np.float64)     # Non-lasing energy dispersion vs. time (MeV)
        powerECOM = np.zeros((n_pulses,t.size), dtype=np.float64)        # Power, reconstructed via ECOM (GW)
        powerERMS = np.zeros((n_pulses,t.size), dtype=np.float64)        # Power, reconstructed via ERMS (GW)
        powerrawECOM=np.zeros((n_pulses,t.size), dtype=np.float64)       # Raw power (no gas detector normalization) reconstructed via ECOM (GW)
        powerrawERMS=np.zeros((n_pulses,t.size), dtype=np.float64)       # Raw power (no gas detector normalization) reconstructed via ERMS (A.U.)
        groupnum=np.zeros(n_pulses, dtype=np.int32)                      # Group number indexing the lasing-off reference profile
        
        # Loop over pulses & compute
        for j in range(n_pulses):

            # Time & energy differences
            distT = (imageStats[j]['xCOM']-imageStats[0]['xCOM'])*physicalUnits['xfsPerPix']  # Time difference between pulses -> fs
            distE = (imageStats[j]['yCOM']-imageStats[0]['yCOM'])*physicalUnits['yMeVPerPix'] # Energy difference between pulses -> MeV
            pulse_delay[j] = distT
            pulse_energydiff[j] = distE

            # Current, CoM, & RMS vs. time (pre-interpolation)
            dt_old = physicalUnits['xfs'][1]-physicalUnits['xfs'][0]            # Time increment (pre-interpolation)
            eCurrent = imageStats[j]['xProfile']/(dt_old*1e-15)*Nelectrons      # Current (electrons/s) [n.b. the original xProfile already was normalized]
            eCOMslice = (imageStats[j]['yCOMslice']-imageStats[j]['yCOM'])*physicalUnits['yMeVPerPix'] # Energy CoM vs. time -> MeV        
            eRMSslice=imageStats[j]['yRMSslice']*physicalUnits['yMeVPerPix']                           # Energy dispersion vs. time -> MeV
            
            # Interpolate time - current
            interp = interp1d(
                    physicalUnits['xfs']-distT,
                    eCurrent,
                    kind='linear',
                    fill_value=0,
                    bounds_error=False,
                    assume_sorted=True)
            eCurrent=interp(t)    
                                                       
            # Interpolate time - energy center-of-mass
            interp = interp1d(
                    physicalUnits['xfs']-distT,
                    eCOMslice,
                    kind='linear',
                    fill_value=0,
                    bounds_error=False,
                    assume_sorted=True)
            eCOMslice=interp(t)
                
            # Interpolate time - energy dispersion
            interp=interp1d(
                    physicalUnits['xfs']-distT,
                    eRMSslice,kind='linear',
                    fill_value=0,
                    bounds_error=False,
                    assume_sorted=True)
            eRMSslice=interp(t)        

            # Find best no lasing match
            N_groups = reference_profiles['eCurrent'].shape[1]
            err = np.zeros(N_groups, dtype=np.float64)
            for g in range(N_groups):
                err[g] = np.corrcoef(eCurrent,reference_profiles['eCurrent'][j,g,:])[0,1]**2
            # The index of the most similar is that with a highest correlation, i.e. the last in the array after sorting it
            order = np.argsort(err)
            refInd = order[-1]
            groupnum[j] = refInd

            # Get changes w.r.t. the reference in the pulse time and energy differences
            pulse_delaychange[j] = distT-reference_profiles['distT'][j,refInd]
            pulse_energydiffchange[j] = distE-reference_profiles['distE'][j,refInd]

            # Assign the currents
            lasingECurrent[j,:] = eCurrent
            nolasingECurrent[j,:] = reference_profiles['eCurrent'][j,refInd,:]

            # Prepare current thresholds for the ECOM and ERMS reconstructions
            threshlevel = self.thresh_current
            threshlasing = np.amax(lasingECurrent[j,:])*threshlevel
            threshnolasing = np.amax(nolasingECurrent[j,:])*threshlevel
            indiceslasing = np.where(lasingECurrent[j,:]>threshlasing)
            indicesnolasing = np.where(nolasingECurrent[j,:]>threshnolasing)      
            ind1 = np.amax([indiceslasing[0][0],indicesnolasing[0][0]])
            ind2 = np.amin([indiceslasing[0][-1],indicesnolasing[0][-1]])        
            if ind1>ind2:
                ind1 = ind2

            # Assign output values where the threshold has been met
            lasingECOM[j,ind1:ind2]=eCOMslice[ind1:ind2]
            nolasingECOM[j,ind1:ind2]=reference_profiles['eCOMslice'][j,refInd,ind1:ind2]
            lasingERMS[j,ind1:ind2]=eRMSslice[ind1:ind2]
            nolasingERMS[j,ind1:ind2]=reference_profiles['eRMSslice'][j,refInd,ind1:ind2]

            # Calculate the power using the centers-of-mass and dispersions for each pulses
            powerECOM[j,:] = ((nolasingECOM[j,:]-lasingECOM[j,:])*ECHARGE*1e6)*eCurrent      # (J/s)
            powerERMS[j,:] = (lasingERMS[j,:]**2-nolasingERMS[j,:]**2)*(eCurrent**(2.0/3.0)) # (A.U.)

        # Normalize the power using the gas detector data
        powerrawECOM = powerECOM*1e-9 
        powerrawERMS = powerERMS.copy()
        # Calculate the normalization constants
        # The total energy should be compatible with the energy detected in the gas detector
        eoffsetfactor = (shotToShot['xrayenergy']-(np.sum(powerECOM)*dt*1e-15))/Nelectrons   # (J)
        escalefactor = np.sum(powerERMS)*dt*1e-15                                            # (J)
        # Apply the corrections to each pulse
        # Calculate the final energy distribution & power agreement
        for j in range(n_pulses):
            powerECOM[j,:] = ((nolasingECOM[j,:]-lasingECOM[j,:])*ECHARGE*1e6+eoffsetfactor)*lasingECurrent[j,:]*1e-9     # (GW)
            powerERMS[j,:] = shotToShot['xrayenergy']*powerERMS[j,:]/escalefactor*1e-9                                    # (GW)
            powerAgreement[j] = 1-np.sum((powerECOM[j,:]-powerERMS[j,:])**2)/(np.sum((
                powerECOM[j,:]-np.mean(powerECOM[j,:]))**2)+np.sum((powerERMS[j,:]-np.mean(powerERMS[j,:]))**2))
            pulse_lasingenergyCOM[j]=np.sum(powerECOM[j,:])*dt*1e-15*1e9    # (J)
            pulse_lasingenergyRMS[j]=np.sum(powerERMS[j,:])*dt*1e-15*1e9    # (J)
                        
        # Return
        pulsecharacterization={
            't':t,                                              # Time (fs)
            'powerrawECOM':powerrawECOM,                        # Raw power (no gas detector normalization) reconstructed via ECOM (GW)
            'powerrawERMS':powerrawERMS,                        # Raw power (no gas detector normalization) reconstructed vis ERMS (A.U.)
            'powerECOM':powerECOM,                              # Power, reconstructed via ECOM (GW)
            'powerERMS':powerERMS,                              # Power, reconstructed via ERMS (GW)
            'powerAgreement':powerAgreement,                    # Agreement factor between the two methods
            'pulse_delay':pulse_delay,                          # Time delay of each pulse with respect to the first one (fs)
            'pulse_delaychange':pulse_delaychange,              # Difference in time delay from the non-lasing reference (fs)
            'xrayenergy':shotToShot['xrayenergy'],              # Total x-ray energy measured by the gas detector (J)
            'lasingenergyperpulse_ECOM': pulse_lasingenergyCOM, # Pulse X-ray energy using the center-of-mass approach (J)
            'lasingenergyperpulse_ERMS': pulse_lasingenergyRMS, # Pulse X-ray energy using the energy dispersion approach (J)
            'pulse_energydiff':pulse_energydiff,                # Energy difference of each pulse with respect to the first one (MeV)
            'pulse_energydiffchange':pulse_energydiffchange,    # Difference in energy differences from the non-lasing reference (MeV)
            'lasingECurrent':lasingECurrent,                    # Electron current for the lasing trace vs. time (electrons/s)
            'nolasingECurrent':nolasingECurrent,                # Electron current for the non-lasing trace vs. time (electrons/s)
            'lasingECOM':lasingECOM,                            # Lasing energy centers-of-mass vs. time (MeV)
            'nolasingECOM':nolasingECOM,                        # Non-lasing energy centers-of-mass vs. time (MeV)
            'lasingERMS':lasingERMS,                            # Lasing energy dispersion vs. time (MeV)
            'nolasingERMS':nolasingERMS,                        # Non-lasing energy dispersion vs. time (MeV)
            'N_pulses': n_pulses,                               # Number of pulses in the shot
            'groupnum': groupnum                                # Group number indexing the lasing-off reference profile
            }
        return pulsecharacterization

    def get_power_profiles(self, calc_agreement_stats=True):
        """ Reconstruct power profiles for the dataset.
        """
        # Get power reconstructions
        self._recons = [None for i in range(len(self._times_good))]
        for idx_good,idx in enumerate(self._progress(self._times_good,desc="Reconstructing power profiles")):
            self.recons[idx_good] = self.get_single_shot_power_profile(idx_good)
        # Get agreement factor statistics
        if calc_agreement_stats:
            self.calculate_agreement_statistics()
        pass

    def calculate_agreement_statistics(self):
        """ Calculate statistics for agreement factors between the reconstruction methods.
        """
        # Tabulate agreement factors
        agreement_factors = np.empty((self.n_pulses,len(self._times_good)))
        for idx in range(len(self._times_good)):
            recon = self.recons[idx]
            for jdx in range(self.n_pulses):
                agreement_factors[jdx,idx] = recon['powerAgreement'][jdx]
        agreement_factor_means = np.mean(agreement_factors,axis=0)
        # Cumulative agreement fractions
        vals = np.arange(-1,1.001,0.01)
        nums = [np.zeros(len(vals),dtype=int) for i in range(self.n_pulses)]
        nums_mean = np.zeros(len(vals),dtype=int)
        cumsums = [np.zeros(201) for i in range(self.n_pulses)]
        cumsum_mean = np.zeros(201)
        for i,val in enumerate(self._progress(vals,desc='Calculating reconstruction agreement statistics')):
            for j in range(self.n_pulses):
                indices = np.where(agreement_factors[j]>=val)[0]
                cumsums[j][i] = len(indices)
            indices_mean = np.where(agreement_factor_means>=val)[0]
            cumsum_mean[i] = len(indices_mean)
        # Order shots by agreement
        inds_ordered = [np.argsort(agreement_factors[i])[::-1] for i in range(self.n_pulses)]
        inds_ordered_means = np.argsort(agreement_factor_means)[::-1]
        # Finish
        self._agreement_stats = {
            'A' : agreement_factors,
            'A_mean' : agreement_factor_means,
            'cumsums' : cumsums,
            'cumsum_means' : cumsum_mean,
            'cumsum_xbins' : np.arange(201),
            'cumsum_xvals' : vals,
            'cumsum_xticks' : np.arange(0,201,25),
            'cumsum_xticklabels' : [f"{int(round(i))}" if np.isclose(i,round(i)) else f"{i:.10f}".rstrip("0") for i in np.arange(-1,1.01,.25)],
            'cumsum_xticklabels_percentages' : np.arange(-100,101,25),
            'inds_ordered' : inds_ordered,
            'inds_ordered_means' : inds_ordered_means,
        }
        pass


    ### Visualization ###

    def show_pulse_power_reconstruction(self,recon, xlim=None, ylim=None,
        figsize=(10,3), title=None, figax=None, returnfig=False, _supress_legend=False):
        """ Display the power profile described by `recon`.
        """
        # Get info
        t = recon['t']
        powerECOM = recon['powerECOM']
        powerERMS = recon['powerERMS']
        agreement = recon['powerAgreement']
        n_pulses = self.n_pulses

        # Show
        if figax is None:
            fig,axs = plt.subplots(1,n_pulses,figsize=figsize)
        else:
            fig,axs = figax
        for i in range(n_pulses):
            ax = axs[i]
            ax.plot(t,powerECOM[i],label='CoM')
            ax.plot(t,powerERMS[i],label='RMS')
            if xlim is not None: ax.set_xlim(xlim)
            if ylim is not None: ax.set_ylim(ylim)
            ax.set_title(f'Pulse {i+1}',size=15)
            ax.set_xlabel('Time (A.U.)')
            ax.set_ylabel('Power (A.U.)')
            ax.text(0.02,0.98,f'A={agreement[i]:.2f}',transform=ax.transAxes,va='top',ha='left',size=13)
            #ax.text(-39,0.5,f'A={agreement[0]:.2f}',transform=ax.transData,va='bottom',ha='left',size=13)
        if title is not None:
            plt.suptitle(title, size=16)
        if not _supress_legend:
            plt.legend()
            plt.tight_layout()
        if returnfig:
            return fig,axs
        else:
            plt.show()

    def show_power_profiles(self, idx, xlim=None, ylim=None, figsize=(10,3),
        title=None, figax=None, returnfig=False, _supress_legend=False):
        """ Display the power profile at valid data index `idx` 
        """
        recon = self.recons[idx]
        fig,axs = self.show_pulse_power_reconstruction(
            recon=recon,
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            title=title,
            figax=figax,
            returnfig=True,
            _supress_legend=_supress_legend,
        )
        if returnfig:
            return fig,axs
        else:
            plt.show()

    def show_power_profile_group(self, indices, xlim=None, ylim=None,
        single_figsize=(10,3), title=None, returnfig=False):
        """ Display a set of power profiles at valid data `indices`
        """
        n_shots_plot = len(indices)

        # Setup figure
        fig,axs = plt.subplots(n_shots_plot,2,figsize=(
            single_figsize[0],single_figsize[1]*n_shots_plot))

        # Loop
        for i,idx in enumerate(indices):
            
            _,_ = self.show_power_profiles(
                idx,
                xlim=xlim,
                ylim=ylim,
                figax=(fig,axs[i]),
                returnfig=True,
                _supress_legend=True
            )

        plt.suptitle(title, size=16)
        plt.legend()
        plt.tight_layout()
        if returnfig:
            return fig,axs
        else:
            plt.show()

    def show_best_power_profiles(self, idx, xlim=None, ylim=None, figsize=(10,3),
        title=None, figax=None, returnfig=False, _supress_legend=False):
        """ Show the idx'th best power profile
        """
        fig,ax = self.show_power_profiles(
            self.agreement_stats['inds_ordered_means'][idx],
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            title=title,
            figax=figax,
            _supress_legend=_supress_legend,
            returnfig=True,
        )
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_worst_power_profiles(self, idx, xlim=None, ylim=None, figsize=(10,3),
        title=None, figax=None, returnfig=False, _supress_legend=False):
        """ Show the idx'th worst power profile
        """
        fig,ax = self.show_power_profiles(
                self.agreement_stats['inds_ordered_means'][::-1][idx],
            xlim=xlim,
            ylim=ylim,
            figsize=figsize,
            title=title,
            figax=figax,
            _supress_legend=_supress_legend,
            returnfig=True
        )
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_best_power_profile_group(self, indices, xlim=None, ylim=None,
        single_figsize=(10,3), title=None, returnfig=False):
        """ Show the indices'th best power profiles
        """
        n_shots_plot = len(indices)

        # Setup figure
        fig,axs = plt.subplots(n_shots_plot,2,figsize=(
            single_figsize[0],single_figsize[1]*n_shots_plot))

        # Loop
        for i,idx in enumerate(indices):
            
            _,_ = self.show_best_power_profiles(
                idx,
                xlim=xlim,
                ylim=ylim,
                figax=(fig,axs[i]),
                returnfig=True,
                _supress_legend=True
            )

        plt.suptitle(title, size=16)
        plt.legend()
        plt.tight_layout()
        if returnfig:
            return fig,axs
        else:
            plt.show()

    def show_worst_power_profile_group(self, indices, xlim=None, ylim=None,
        single_figsize=(10,3), title=None, returnfig=False):
        """ Show the indices'th worst power profiles
        """
        n_shots_plot = len(indices)

        # Setup figure
        fig,axs = plt.subplots(n_shots_plot,2,figsize=(
            single_figsize[0],single_figsize[1]*n_shots_plot))

        # Loop
        for i,idx in enumerate(indices):
            
            _,_ = self.show_worst_power_profiles(
                idx,
                xlim=xlim,
                ylim=ylim,
                figax=(fig,axs[i]),
                returnfig=True,
                _supress_legend=True
            )

        plt.suptitle(title, size=16)
        plt.legend()
        plt.tight_layout()
        if returnfig:
            return fig,axs
        else:
            plt.show()

    def show_agreement_histograms(self,figsize=(12,3),nbins=50,title=None,returnfig=False):
        """
        """
        bins = np.linspace(-1,1,nbins)
        fig,axs = plt.subplots(1,self.n_pulses+1,figsize=(12,3))
        for i in range(self.n_pulses):
            axs[i].hist(self.agreement_stats['A'][i],bins=bins)
            axs[i].set_title(f"Pulse {i+1}", size=14)
            axs[i].set_xlabel('A (unitless)')
            axs[i].set_ylabel('Counts')
        ax_mean = axs[self.n_pulses]
        ax_mean.hist(self.agreement_stats['A_mean'],bins=bins)
        ax_mean.set_title('Mean', size=14)
        ax_mean.set_ylabel('Counts')
        title = title if title is not None else f"Agreement factor histograms over {len(self._times_good)} shots"
        plt.suptitle(title, size=16)
        plt.tight_layout()
        if returnfig:
            return fig,axs
        else:
            plt.show()

    def show_cumulative_histograms(self,figsize=(12,3),title=None,percent=False,returnfig=False):
        """
        """
        # Get info
        x = self.agreement_stats['cumsum_xbins']
        cumsums = self.agreement_stats['cumsums']
        cumsums_means = self.agreement_stats['cumsum_means']
        norm = np.max(cumsums_means)
        xvals = self.agreement_stats['cumsum_xvals']
        xticks = self.agreement_stats['cumsum_xticks']
        xticklabels = self.agreement_stats['cumsum_xticklabels'] if not percent else self.agreement_stats['cumsum_xticklabels_percentages']
        xaxis_label = "A (%)" if percent else "A (unitless)"
        fig,axs = plt.subplots(1,self.n_pulses+1,figsize=figsize)
        for i in range(self.n_pulses):
            axs[i].bar(
                x=x,
                height=cumsums[i]/norm,
                width=1,
                align='edge'
            )
            axs[i].set_title(f'Pulse {i+1}', size=14)
            axs[i].set_xlabel(xaxis_label)
            axs[i].set_ylabel('Counts')
            axs[i].set_ylim(0,1)
            axs[i].set_xlim(x[0],x[-1])
            axs[i].set_xticks(xticks)
            axs[i].set_xticklabels(xticklabels)
            axs[i].vlines(100,0,axs[i].get_ylim()[1],color='k',lw=0.5,ls='--')
        ax_mean = axs[self.n_pulses]
        ax_mean.bar(
            x=x,
            height=cumsums_means/norm,
            width=1,
            align='edge'
        )
        ax_mean.set_title('Mean', size=14)
        ax_mean.set_xlabel(xaxis_label)
        ax_mean.set_ylabel('Counts')
        ax_mean.set_ylim(0,1)
        ax_mean.set_xlim(x[0],x[-1])
        ax_mean.set_xticks(xticks)
        ax_mean.set_xticklabels(xticklabels)
        ax_mean.vlines(100,0,ax_mean.get_ylim()[1],color='k',lw=0.5,ls='--')
        title = title if title is not None else f'Counts with A > % over {len(self._times_good)} valid shots'
        plt.suptitle(title ,size=16)
        plt.tight_layout()
        if returnfig:
            return fig,axs
        else:
            plt.show()

