# Lasing-off Reference Class

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from matplotlib.cm import viridis
from scipy.interpolate import interp1d
from xtcav_processor import XTCAVProcessor, ECHARGE

class XTCAVLasingOffReference(XTCAVProcessor):

    def __init__(self, data_source, shots_per_group=100, env=None, iteration='frames', verbose=True, _test_xy=False, _preindex=False):
        """
        Parameters
        ----------
        data_source : a psana DataSource instance
            Must point to a dark run
        shots_per_group : integer
            Number of shots to include in each average profile
        env : None or a psana.Env
            if None uses dataSouce.env()
        iteration : string in ('all', 'good', 'frames'):
            Determines which data is traversed when using the shot iterator.
            'all' iterates over all shots. 'good' iterates over shots with all data
            present.  'frames' iterates over all data with XTCAV camera data present.
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
            verbose=verbose,
            _test_xy=_test_xy,
            _preindex=_preindex,
        )
        self._shots_per_group = shots_per_group
        self._reference_profiles = None
        pass

    @property
    def shots_per_group(self):
        return self._shots_per_group
    
    @property
    def reference_profiles(self):
        return self._reference_profiles


    def calculate_reference_profiles(self, returnans=False):
        """
        Calculate the average profile from a non-lasing data run.
        Requires .get_shot_by_shot_statistics has been run.

        Returns
        -------
        (dictionary) : the averaged reference profiles, in the following keys/values:

            Key             Value Shape            Value
            ---             -----------            -----
            't'             (T,)                   time axis (fs)
            'eCurrent'      (N_pu,N_pr,T)          current (#electrons/s)
            'eCOMslice'     (N_pu,N_pr,T)          energy center of masses vs. time (MeV)
            'eRMSslice'     (N_pu,N_pr,T)          energy dispersion vs. time (MeV)
            'distT'         (N_pu,N_pr)            time delay between pulse centers of masses with respect to the first pulse (fs)
            'distE'         (N_pu,N_pr)            energy difference between pulse centers of masses with respect to the center of the first pulse (MeV)
            'tRMS'          (N_pu,N_pr)            mean time dispersion (fs)
            'eRMS'          (N_pu,N_pr)            mean energy dispersion (MeV)
            'N_pulses'      int                    number of pulses
            'N_puofiles'    int                    number of average profiles
            'N_per_profile' int                    number of profiles averaged in each group
            # TODO - add time/fiducials
            ##'eventTime'   (N_pu,N_pr)            unix times used for jumping to events
            ## 'eventFid'   (N_pu,N_pr)            fiducial values used for jumping to events
        """
        # Validate & assign inputs
        assert(self._pulse_statistics is not None), "Pulse statistics not found; run .get_shot_by_shot_statistics()!"
        assert(self._shot_to_shot_params is not None), "Shot-by-shot parameters not found; run .get_shot_by_shot_statistics()!"
        assert(self._physical_units is not None), "Physical units not found; run .get_shot_by_shot_statistics()!"
        list_image_stats = self._pulse_statistics
        list_shot_to_shot_params = self._shot_to_shot_params
        list_physical_units = self._physical_units
        shots_per_group = self.shots_per_group

        # Get sizes
        N = len(list_image_stats[0])                  # number of shots
        N_pulses = len(list_image_stats)              # pulses per shot
        N_groups = int(np.floor(N/shots_per_group))
        assert(N == len(list_shot_to_shot_params) == len(list_physical_units)), "pulse info dict lengths are not consistent!"

        # Set a time vector   
        # initialize values (maximum,  minimum, & increment)
        t_max = 0
        t_min = 0
        t_incr = 1000
        # find values, looping through physical units by shot
        for i in range(N):
            t_max = np.amax([t_max,np.amax(list_physical_units[i]['xfs'])])
            t_min = np.amin([t_min,np.amin(list_physical_units[i]['xfs'])])
            t_incr = np.amin([t_incr,np.abs(list_physical_units[i]['xfsPerPix'])])
        # find the number of electrons in each shot
        N_electrons = np.zeros(N, dtype=np.float64);
        for i in range(N): 
            N_electrons[i] = list_shot_to_shot_params[i]['dumpecharge']/ECHARGE      
        # set the time vector increment to half the minimum pixel step size
        dt = t_incr/2
        # create the master time vector in fs
        t = np.arange(t_min,t_max+dt,dt)

        # Make containers
        # First index = pulse number
        # Second index = group number
        average_current = np.zeros((N_pulses,N_groups,len(t)), dtype=np.float64)    # Electron current (#electrons/s)
        average_eCOMslice = np.zeros((N_pulses,N_groups,len(t)), dtype=np.float64)  # Energy center of masses for each time in MeV
        average_eRMSslice = np.zeros((N_pulses,N_groups,len(t)), dtype=np.float64)  # Energy dispersion for each time in MeV
        average_DT = np.zeros((N_pulses,N_groups), dtype=np.float64)                # time delay between pulse centers of masses with respect to the first pulse (fs)
        average_DE = np.zeros((N_pulses,N_groups), dtype=np.float64)                # energy difference between pulse centers of masses with respect to the center of the first pulse (MeV)
        average_tRMS = np.zeros((N_pulses,N_groups), dtype=np.float64)              # Total dispersion in time in fs
        average_eRMS = np.zeros((N_pulses,N_groups), dtype=np.float64)              # Total dispersion in energy in MeV
        ##event_time = np.zeros((N_pulses,N_groups), dtype=np.uint64)
        ##event_fid = np.zeros((N_pulses,N_groups), dtype=np.uint32)

        # Progress bars
        progbar1 = tqdm(desc=f"Looping through the {N_pulses} pulses", total=N_pulses)
        progbar2 = tqdm(desc="Interpolating XTCAV profiles", total=N)
        progbar3 = tqdm(desc=f"Forming {N_groups} groups & calculating averages", total=N_groups)
        
        # Treat each pulse separately
        for j in range(N_pulses):
            progbar1.n = j
            progbar1.refresh()
            
            # get pulse stats
            image_stats = list_image_stats[j]
            
            # Get interpolated time profiles
            progbar2.n=0
            progbar2.refresh()
            profiles_t = np.zeros((N,len(t)), dtype=np.float64)
            for i in range(N):
                dist_t = (image_stats[i]['xCOM']-list_image_stats[0][i]['xCOM'])*list_physical_units[i]['xfsPerPix']
                profiles_t[i,:] = interp1d(
                    list_physical_units[i]['xfs']-dist_t,image_stats[i]['xProfile'],
                    kind='linear',
                    fill_value=0,
                    bounds_error=False,
                    assume_sorted=True
                )(t)
                if i%10==0:
                    progbar2.n = i+1
                    progbar2.refresh()
            progbar2.n=N
            progbar2.refresh()

            ### Organize the pulses into groups

            # loop, assigning shots/pulses to groups
            group = np.full(N, -1, dtype=np.int32) # indicates which group each pulse is in; -1="unassigned"
            for g in range(N_groups):
                progbar3.n=g
                progbar3.refresh()
                # find the first unassigned pulse and assign it to this group
                currRef = np.where(group==-1)[0][0]
                group[currRef]=g
                # calculate the correlation of the current profile to the rest of available profiles
                err = np.zeros(N,dtype=np.float64)        
                for i in range(currRef,N): 
                    if group[i] == -1:
                        err[i] = np.corrcoef(profiles_t[currRef,:],profiles_t[i,:])[0,1]**2
                # the 'shotsPerGroup-1' profiles with the highest correlation will be also assigned to the same group
                order=np.argsort(err)            
                for i in range(0,shots_per_group-1): 
                    group[order[-(1+i)]] = g

            # calculate averages
            for g in range(N_groups):
                # loop through profiles in this group
                group_indices = np.where(group==g)[0]
                for i in group_indices:
                # for i in range(N):    
                #     if group[i]==g:

                    # TODO - add time/fiducials
                    ##event_time[j][g] = list_shot_to_shot_params[i]['unixtime']
                    ##event_fid[j][g] = list_shot_to_shot_params[i]['fiducial']
                    dist_t = (image_stats[i]['xCOM']-list_image_stats[0][i]['xCOM'])*list_physical_units[i]['xfsPerPix']  # time - pixels -> fs
                    dist_e = (image_stats[i]['yCOM']-list_image_stats[0][i]['yCOM'])*list_physical_units[i]['yMeVPerPix'] # time - pixels -> MeV
                    average_DT[j,g] = average_DT[j,g]+dist_t       # accumulate
                    average_DE[j,g] = average_DE[j,g]+dist_e       # accumulate
                    
                    average_tRMS[j,g] = average_tRMS[j,g]+image_stats[i]['xRMS']*list_physical_units[i]['xfsPerPix']   # convert -> fs and accumulate
                    average_eRMS[j,g] = average_eRMS[j,g]+image_stats[i]['yRMS']*list_physical_units[i]['yMeVPerPix']  # convert -> MeV and accumulate

                    dt_old = list_physical_units[i]['xfs'][1]-list_physical_units[i]['xfs'][0]    # dt before interpolation
                    eCurrent = image_stats[i]['xProfile']/(dt_old*1e-15)*N_electrons[i]   # current (electrons/s)
                    
                    eCOMslice = (image_stats[i]['yCOMslice']-image_stats[i]['yCOM'])*list_physical_units[i]['yMeVPerPix'] # Energy CoM vs. t, -> MeV
                    eRMSslice = image_stats[i]['yRMSslice']*list_physical_units[i]['yMeVPerPix']                                 # Energy dispersion vs. t, -> MeV
                        
                    interp = interp1d(list_physical_units[i]['xfs']-dist_t,eCurrent,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True)  # time interpolation                   
                    average_current[j,g,:] = average_current[j,g,:]+interp(t);  # accumulate
                                                
                    interp = interp1d(list_physical_units[i]['xfs']-dist_t,eCOMslice,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True) # time interpolation
                    average_eCOMslice[j,g,:] = average_eCOMslice[j,g,:]+interp(t);          # accumulate
                    
                    interp = interp1d(list_physical_units[i]['xfs']-dist_t,eRMSslice,kind='linear',fill_value=0,bounds_error=False,assume_sorted=True) # time interpolation
                    average_eRMSslice[j,g,:] = average_eRMSslice[j,g,:]+interp(t);          # accumulate

                # Normalize
                average_current[j,g,:] = average_current[j,g,:]/shots_per_group
                average_eCOMslice[j,g,:] = average_eCOMslice[j,g,:]/shots_per_group   
                average_eRMSslice[j,g,:] = average_eRMSslice[j,g,:]/shots_per_group
                average_DT[j,g] = average_DT[j,g]/shots_per_group
                average_DE[j,g] = average_DE[j,g]/shots_per_group
                average_tRMS[j,g] = average_tRMS[j,g]/shots_per_group
                average_eRMS[j,g] = average_eRMS[j,g]/shots_per_group  

            progbar3.n=N_groups
            progbar3.refresh()

        progbar1.n = N_pulses
        progbar1.refresh()
        progbar2.close()
        progbar3.close()
        progbar1.close()
            
        # Prepare output and return
        self._reference_profiles = {
            't':t,                            # time (fs)
            'eCurrent':average_current,       # current in (#electrons/s)
            'eCOMslice':average_eCOMslice,    # energy center of masses vs. time (MeV)
            'eRMSslice':average_eRMSslice,    # energy dispersion vs. time (MeV)
            'distT':average_DT,               # time delay between pulse centers of masses with respect to the first pulse (fs)
            'distE':average_DE,               # energy difference between pulse centers of masses with respect to the center of the first pulse (MeV)
            'tRMS': average_tRMS,             # mean time dispersion (fs)
            'eRMS': average_eRMS,             # mean energy dispersion (MeV)
            'N_pulses': N_pulses,                   # number of bunches
            'N_profiles': N_groups,                   # number of groups
            'N_per_profile': shots_per_group, # number of profiles averaged in each group
            # TODO - add time/fiducials
            ##'eventTime': event_time,        # unix times used for jumping to events
            ##'eventFid': event_fid           # fiducial values used for jumping to events
            }
        if returnans:
            return self._reference_profiles
        else:
            pass



    def show_reference_profiles(self,idx_pulse=0,lims=[-36,36],cmap=None,returnfig=False):
        """
        """
        # Number of profiles
        N = self.reference_profiles['N_profiles']

        # Colormap
        if cmap is None:
            cmap = viridis
        colors = [cmap(i/float(N-1)) for i in range(N)]

        # Get profile info
        t = self.reference_profiles['t']
        currents = [self.reference_profiles['eCurrent'][idx_pulse,i] for i in range(N)]
        CoMs = [self.reference_profiles['eCOMslice'][idx_pulse,i] for i in range(N)]
        RMSs = [self.reference_profiles['eRMSslice'][idx_pulse,i] for i in range(N)]

        # Plot
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(9,9))
        for i in range(N):
            ax1.plot(t,currents[i],color=colors[i])
            ax2.plot(t,CoMs[i],color=colors[i])
            ax3.plot(t,RMSs[i],color=colors[i])
        ax1.set_xlabel('Time (A.U.)')
        ax2.set_xlabel('Time (A.U.)')
        ax3.set_xlabel('Time (A.U.)')
        ax1.set_ylabel('Current (A.U.)')
        ax2.set_ylabel('Energy CoM (A.U.)')
        ax3.set_ylabel('Energy RMS (A.U.)')
        ax1.set_xlim(lims[0],lims[1])
        ax2.set_xlim(lims[0],lims[1])
        ax3.set_xlim(lims[0],lims[1])
        ax1.set_title('Current vs. Time',size=15)
        ax2.set_title('Energy Center-of-Mass vs. Time',size=15)
        ax3.set_title('Energy Dispersion vs. Time',size=15)
        plt.suptitle(r'Lasing-off Reference Profiles (UNCALIBRATED)',size=18)
        plt.tight_layout()
        
        # Return
        if returnfig:
            return fig,(ax1,ax2,ax3)
        else:
            plt.show()


    def show_reference_profiles_boxplots(self,idx_pulse=0,lims=[-36,36],returnfig=False):
        """
        """
        # Number of profiles
        N = self.reference_profiles['N_profiles']

        # Profile info
        t = self.reference_profiles['t']
        currents = [self.reference_profiles['eCurrent'][idx_pulse,:,i] for i in range(len(t))]
        CoMs = [self.reference_profiles['eCOMslice'][idx_pulse,:,i] for i in range(len(t))]
        RMSs = [self.reference_profiles['eRMSslice'][idx_pulse,:,i] for i in range(len(t))]

        # Lims
        # TODO: will need to be modified if we want ticks at a spacing other than
        # every 10 units
        xmi,xma = lims
        sxmi,sxma = np.sign(lims[0]),np.sign(lims[1])
        xmin = int(np.floor((sxmi*lims[0])//10))*10*sxmi
        xmax = int(np.floor((sxma*lims[1])//10))*10*sxma
        xticks = np.arange(xmin,xmax+1,10)

        # Plot
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(9,9))
        ax1.boxplot(currents,positions=t)
        ax2.boxplot(CoMs,positions=t)
        ax3.boxplot(RMSs,positions=t)
        ax1.set_xlim(lims[0],lims[1])
        ax2.set_xlim(lims[0],lims[1])
        ax3.set_xlim(lims[0],lims[1])
        ax1.set_xticks(xticks)
        ax2.set_xticks(xticks)
        ax3.set_xticks(xticks)
        ax1.set_xticklabels(xticks)
        ax2.set_xticklabels(xticks)
        ax3.set_xticklabels(xticks)
        ax1.set_xlabel('Time (A.U.)')
        ax2.set_xlabel('Time (A.U.)')
        ax3.set_xlabel('Time (A.U.)')
        ax1.set_ylabel('Current (A.U.)')
        ax2.set_ylabel('Energy CoM (A.U.)')
        ax3.set_ylabel('Energy RMS (A.U.)')
        ax1.set_title('Current vs. Time',size=15)
        ax2.set_title('Energy Center-of-Mass vs. Time',size=15)
        ax3.set_title('Energy Dispersion vs. Time',size=15)
        plt.suptitle(r'Lasing-off Reference Profiles (UNCALIBRATED)',size=18)
        plt.tight_layout()

        # Return
        if returnfig:
            return fig,(ax1,ax2,ax3)
        else:
            plt.show()



