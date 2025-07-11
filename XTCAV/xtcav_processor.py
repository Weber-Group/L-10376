# Defines a class for xtcav processing
# Builds from and synthesizes older codes

import warnings
import numpy as np
from tqdm.auto import tqdm
from math import degrees
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.ndimage import binary_closing, median_filter
from skimage.measure import label
import psana

# Globals
#ECHARGE = 1.60217657e-19
ECHARGE = 1.602176634e-19   # corrected for 2019 revision of SI units

class XTCAVProcessor(object):
    """
    An XTCAV data processing class.  It's purpose is to build on and synthesize
    existing XTCAV analysis codes developed at SLAC.

    Instantiation & Data Access
    ---------------------------
    For some psana.DataSource `ds`,
    
    >>> xtpr = XTCAVProcessor(ds)

    will instantiate an XTCAV processing object instance,

    >>> i,j,evt = next(xtpr.shot_iterator)

    will grab the first 'ok' event (i.e. containing data from the gas detector,
    electron beam, and XTCAV camera), where `i` is the 'ok' event index (i.e.
    in this example `i` is 0 indicating the first 'ok' event), `j` is the event
    index (i.e. if the first 'ok' event found in this data run is the 150'th
    shot, `j` is 149) and `evt` is a psana.Event instance. Subsequently running

    >>> xtpr.set_current_event(evt)
    >>> xtpr.show_current_frame(
    >>>     vrange=(0,350),
    >>>     bins=np.linspace(0,1000,100),
    >>>     fov=fov
    >>> )

    will set and then display data from this event, showing the XTCAV camera
    frame, a zoom-in on the shot(s), and an intensity histogram.

    >>> for i,j,evt in xtpr.shot_iterator:
    >>>     # your code
    >>>     pass

    will iterate over events all events in the data source. Once you have an
    event, you can access data one of two ways - either call
    
    >>> ebeam, gas, image = xtpr.get_data(event)

    to get the data directly, or call

    >>> xtpr.set_current_event(event)

    to "set" the event, at which point the data for that event is accessible as
    properties at `.ebeam`, `.gasdetector`, and `.frame`.  Data not present for
    this event will return None. So

    >>> i,j,evt = next(xproc.shot_iterator)
    >>> xtpr.set_current_event(evt)
    >>> plt.imshow(xtpr.frame)

    will display the image captured on the XTCAV camera for the j'th event,
    provided that that data exists.

    Event Iterators
    ---------------

    >>> xtpr.shot_iterator

    is a Python iterator which loops through events.  The iterator can be reset
    to the beginning of the dataset with

    >>> xtpr.reset_shot_iterator()

    Instantiating with the `iteration` keyword enables looping over a different
    event set, i.e. setting

    >>> xtpr = XTCAVProcessor(ds, iteration='all')
    >>> xtpr = XTCAVProcessor(ds, iteration='frames')

    iterates instead over all events regardless of what data is present in the
    first case or all events containing image frames on the XTCAV camera in the
    second case.  This can also be reset post-instantiation with

    >>> xtpr.reset_iteration_type(iteration=*)

    for an appropriate value of `*`.

    The  XTCAVProcessor instance will keep track of which events have which data
    types present, and will have created a complete log after it finishes one
    complete iteration over the dataset.  At this point you can see the event
    indices containing various data subsets with

    >>> xtpr._times_ok   # to see indices of events with all data present
    >>> xtpr._times_*    # to see indices of events with various data combos present

    where * can be (ebeam, gasdetector, frames, ebeamAndGas, ..., ebeam_only,
    ..., none). You can now iterate over a data subset of interest with, e.g.

    >>> for i,j,evt in xtpr.generate_ebeam_shots_iterator():

    to iterate over all events with ebeam data, or you can simply reset the
    default iterator with `xtpr.reset_shot_iterator()`.

    Finally, 'good' shots have all detectors present *and* their statistics
    can be calculated.  These are shots that meet the 'ok' criteria and
    additionally have all their shot-to-shot metadata, have no saturated
    pixels on the camera, the pulses are not too close to the camera edges,
    pulses can be successfully split into distinct frames for multi-pulse data,
    and physical units can be calibrated.  'good' shots are determined when
    `xtpr.get_shot_by_shot_statistics` is run. At this point the shot
    iterator is automatically reset to iterate over 'good' shots.


    (Benjamin H. Savitzky, 6/2025)

    """

    def __init__(self, data_source, env=None, iteration='ok', fov=(120,240),
        dark_reference=None, n_pulses=1, verbose=True, _test_xy=False,
        _preindex=False, _patch_shot_metadata=False,
        _overwrite_global_metadata={}, _overwrite_shotByShot_metadata={}):
        """
        Parameters
        ----------
        data_source : a psana DataSource instance
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
        n_pulses : integer
            The number of pulses per shot
        verbose : bool
            Toggle vebosity
        _test_xy : bool
            Internal testing flag
        _preindex : bool
            Toggles indexing events on instantiation. If True, loops through all
            events and determines which data types (ebeam, gas detector, XTCAV
            camera frame) are present for each event.
        _patch_shotMetadata : bool
            If True, replace shot RF amp with global RF amp. Patch for exp. L10376-23
            where all shot-to-shot RF amp and RF phase data was recorded as 0.
        _overwrite_global_metadata : dict
            Dictionary items will be used in place of EPICS data of the same name
            when assigning global metadata. Valid keys are ('umperpix',
            'strstrength','rfampcalib','rfphasecalib','dumpe','dumpdisp')
        _overwrite_local_metadata : dict
            Dictionary items will be used in place of shot-by-shot metadata of the
            same name. Valid keys are ('ebeamcharge','dumpecharge','xtcavrfamp',
            'xtcavrfphase')
        """
        self._verbose = verbose
        self._test_xy = _test_xy
        self._patch_shot_metadata = _patch_shot_metadata
        self.data_source = data_source
        self._set_env(env)
        self._setup_calibration_metadata_overwriting(
            _overwrite_global_metadata,
            _overwrite_shotByShot_metadata,
        )
        ok = self._setup_global_calibrations()
        if not ok:
        #    raise Exception('Failed to find all global calibrations!')
            warnings.warn('Failed to find all global calibrations!')
        self._set_run()
        self._set_detectors()
        self._set_iteration_type(iteration)
        if _preindex:
            self._preindex_shots()
            self.reset_shot_iterator()
        else:
            self._data_is_indexed = False
            self.reset_shot_iterator()
        self._data_stats_are_calculated = False
        self._setup_vis_params()
        self._update_vis_params(fov=fov)
        self._setup_denoise_params()
        if dark_reference is not None:
            self.set_darkreference(dark_reference)
        self._set_n_pulses(n_pulses)
        pass

    ### Setup methods ###

    def _set_env(self, env):
        # Setup env
        if env is None:
            self._env = self.data_source.env()
        else:
            self._env = env
        # Setup epics store
        self._epics_store = self._env.epicsStore()
        pass

    def _setup_calibration_metadata_overwriting(
        _overwrite_global_metadata,
        _overwrite_shotByShot_metadata,
        ):
        """
        """
        # Global items
        valid_keys_global = (
            'umperpix',
            'strstrength',
            'rfampcalib',
            'rfphasecalib',
            'dumpe',
            'dumpdisp'
        )
        self._global_calibration_overwrites = {}
        for k,v in _overwrite_global_metadata.items():
            if k not in valid_keys_global:
                warnings.warn(f"Key {k} is not a valid global metadata key overwrite; must be in {valid_keys_global}")
            else:
                self._global_calibration_overwrites[k] = v

        # Shot-by-shot items
        valid_keys_shotByShot = (
            'ebeamcharge',
            'dumpecharge',
            'xtcavrfamp',
            'xtcavrfphase',
        )
        self._shotByShot_calibration_overwrites = {}
        for k,v in _overwrite_shotByShot_metadata.items():
            if k not in valid_keys_shotByShot:
                warnings.warn(f"Key {k} is not a valid shot-by-shot metadata key overwrite; must be in {valid_keys_shotByShot}")
            else:
                self._shotByShot_calibration_overwrites[k] = v
        pass


    def _setup_global_calibrations(self):
        """
        Setup the global XTCAV calibration parameters from the epicsStore

        Returns
        -------
        (bool) True iff all the data was retrieved correctly
        """
        # Global calibratio values
        ok = [1]
        if 'umperpix' not in self._global_calibration_overwrites.keys():
            umperpix = self._get_global_calibration_value(['XTCAV_calib_umPerPx','OTRS:DMP1:695:RESOLUTION'],ok)
        else:
            umperpix = self.global_calibration_overwrites['umperpix']
        if 'strstrength' not in self._global_calibration_overwrites.keys():
            strstrength = self._get_global_calibration_value(['XTCAV_strength_par_S','Streak_Strength','OTRS:DMP1:695:TCAL_X'],ok)
        else:
            strstrength = self.global_calibration_overwrites['strstrength']
        if 'rfampcalib' not in self._global_calibration_overwrites.keys():
            rfampcalib = self._get_global_calibration_value(['XTCAV_Amp_Des_calib_MV','XTCAV_Cal_Amp','SIOC:SYS0:ML01:AO214'],ok)
        else:
            rfampcalib = self.global_calibration_overwrites['rfampcalib']
        if 'rfphasecalib' not in self._global_calibration_overwrites.keys():
            rfphasecalib = self._get_global_calibration_value(['XTCAV_Phas_Des_calib_deg','XTCAV_Cal_Phase','SIOC:SYS0:ML01:AO215'],ok)
        else:
            rfphasecalib = self.global_calibration_overwrites['rfphasecalib']
        if 'dumpe' not in self._global_calibration_overwrites.keys():
            dumpe = self._get_global_calibration_value(['XTCAV_Beam_energy_dump_GeV','Dump_Energy','REFS:DMP1:400:EDES'],ok)
        else:
            dumpe = self.global_calibration_overwrites['dumpe']
        if 'dumpdisp' not in self._global_calibration_overwrites.keys():
            dumpdisp = self._get_global_calibration_value(['XTCAV_calib_disp_posToEnergy','Dump_Disp','SIOC:SYS0:ML01:AO216'],ok)
        else:
            dumpdisp = self.global_calibration_overwrites['dumpdisp']

        self._global_calibrations = {
            'umperpix':umperpix,            # Pixel size of the XTCAV camera
            'strstrength':strstrength,      # Strength parameter
            'rfampcalib':rfampcalib,        # Calibration of the RF amplitude
            'rfphasecalib':rfphasecalib,    # Calibration of the RF phase
            'dumpe':dumpe,                  # Beam energy: dump config
            'dumpdisp':dumpdisp             # Vertical position to energy: dispersion
            }
        
        # Camera saturation value
        if self._epics_store.value('XTCAV_Analysis_Version') is not None:
            self._camera_saturation_value = (1<<12)-1  # bit depth 12
        else:
            self._camera_saturation_value = (1<<14)-1  # bit depth 14

        # Set up containers for statistics
        self._pulse_statistics = None
        self._shot_to_shot_params = None
        self._physical_units = None

        # Return
        return ok[0]

    def _get_global_calibration_value(self,names,ok):
        """
        Get the calibration value associated with `names`. Iterates through
        the list of names, returning the first value found.

        Parameters
        ----------
        names : list of strings
            String values should be psana.epicsStore names or aliases
        ok : length 1 list of ints
            Flag indicating success or failure. Updated in place (not returned)

        Returns
        -------
        Value of the calibration variable. If not found, returns None.
        """
        for n in names:
            val = self._epics_store.value(n)
            if val is not None: return val
        warnings.warn_explicit(f'No XTCAV Calibration for epics variable {names[0]}',UserWarning,'XTCAV',0)
        ok[0]=0  # notify caller that no value was found for a variable
        return 0

    def _set_run(self):
        """ Assumes data_source has a single run
        """
        if self._verbose:
            print("Setting up data run times...")
        self._run = next(self.data_source.runs())
        self._times = self._run.times()
        if self._verbose:
            print("Done.")

    def _setup_vis_params(self):
        """ Initialize visualization parameters
        """
        self._vis_params = {
            "vrange" : None,
            "fov" : None,
            "bins" : None,
            "preprocess" : False,
        }
    def _update_vis_params(self, **p):
        """
        """
        new_params = {}
        keys = (
            'vrange',
            'fov',
            'bins',
            'preprocess',
        )
        for k in keys:
            if k in p.keys():
                if p[k] is not None:
                    new_params[k] = p[k]
        self._vis_params = self._vis_params | new_params

    def _setup_denoise_params(self):
        """ Initialize denoising parameters
        """
        self._denoise_params = {
            "bksb" : True,
            "nstd" : 2,
            "thresh_mode" : "median",
            "close" : 1,
            "noise_x" : 100,
            "noise_y" : 100,
            "noise_x_end" : True,
            "noise_y_end" : True,
            "medfilt_size" : 1,
            "normalize" : True,
        }
        pass
    def _update_denoise_params(self, **p):
        """
        """
        new_params = {}
        keys = (
            'bksb',
            'nstd',
            'thresh_mode',
            'close',
            'noise_x',
            'noise_y',
            'noise_x_end',
            'noise_y_end',
            'medfilt_size',
            'normalize'
        )
        for k in keys:
            if k in p.keys(): new_params[k] = p[k]
        self._denoise_params = self._denoise_params | new_params

    def _set_detectors(self):
        self._set_ebeam_data()
        self._set_gasdetector_data()
        self._set_xtcav_camera()
        self._set_data_source_types_and_getters()
        pass

    def _set_ebeam_data(self):
        #Ebeam type: it should actually be the version 5 which is the one that contains xtcav stuff
        self.ebeam_data=psana.Source('BldInfo(EBeam)')
        self._ebeam=None
        pass

    def _set_gasdetector_data(self):
        #Gas detectors for the pulse energies
        self.gasdetector_data=psana.Source('BldInfo(FEEGasDetEnergy)')
        self._gasdetector=None
        pass

    def _set_xtcav_camera(self):
        #Camera and type for the xtcav images
        self.xtcav_camera = psana.Source('DetInfo(XrayTransportDiagnostic.0:Opal1000.0)')
        self.xtcav_type=psana.Camera.FrameV1
        self._rawimage=None
        pass

    def _set_data_source_types_and_getters(self):
        """ 
        Determine the source types and set according getter methods for
        ebeam and gas detector data
        """
        self._get_ebeam = None
        self._get_gasdetector = None
        self._xtcav_frame_shape = None
        self._xtcav_frame_x = None
        self._xtcav_frame_y = None
        self._xtcav_frame_xx = None
        self._xtcav_frame_yy = None
        iterator = self._progress(self._times,desc="Setting data source types...")
        for i,t in enumerate(self._times):
            if self._verbose:
                iterator.update()
            # set event
            evt = self._run.event(t)
            # set ebeam getter
            if self._get_ebeam is None:
                ebeam = evt.get(psana.Bld.BldDataEBeamV7,self.ebeam_data)
                if ebeam:
                    self.ebeam_type = psana.Bld.BldDataEBeamV7
                    self._get_ebeam = lambda event: event.get(psana.Bld.BldDataEBeamV7,self.ebeam_data)
                else:
                    ebeam = evt.get(psana.Bld.BldDataEBeamV6,self.ebeam_data)
                    if ebeam:
                        self.ebeam_type = psana.Bld.BldDataEBeamV6
                        self._get_ebeam = lambda event: event.get(psana.Bld.BldDataEBeamV6,self.ebeam_data)
                    else:
                        ebeam = evt.get(psana.Bld.BldDataEBeamV5,self.ebeam_data)
                        if ebeam:
                            self.ebeam_type = psana.Bld.BldDataEBeamV5
                            self._get_ebeam = lambda event: event.get(psana.Bld.BldDataEBeamV5,self.ebeam_data)
            # set gas detector getter
            if self._get_gasdetector is None:
                gasdetector=evt.get(psana.Bld.BldDataFEEGasDetEnergy,self.gasdetector_data)
                if gasdetector:
                    self.gasdetector_type = psana.Bld.BldDataFEEGasDetEnergy
                    self._get_gasdetector = lambda event: event.get(psana.Bld.BldDataFEEGasDetEnergy,self.gasdetector_data)
                else:
                    gasdetector=evt.get(psana.Bld.BldDataFEEGasDetEnergyV1,self.gasdetector_data)         
                    if gasdetector:
                        self.gasdetector_type = psana.Bld.BldDataFEEGasDetEnergyV1
                        self._get_gasdetector = lambda event: event.get(psana.Bld.BldDataFEEGasDetEnergyV1,self.gasdetector_data)
            # set xtcav frame size
            if self._xtcav_frame_shape is None:
                frame = self._get_frame(evt)
                if frame is not None:
                    rawimage=frame.data16()
                    if self._test_xy: rawimage = rawimage[:,:-1]
                    self._xtcav_frame_shape = rawimage.shape
                    self._xtcav_frame_x = np.arange(self.nx)
                    self._xtcav_frame_y = np.arange(self.ny)
                    self._xtcav_frame_xx,self._xtcav_frame_yy = np.meshgrid(self.x,self.y)
            # if we have both getters, break and return
            if self._get_ebeam is not None and self._get_gasdetector is not None and self._xtcav_frame_shape is not None:
                # finish progress bar
                if self._verbose:
                    iterator.n = iterator.total
                    iterator.update()
                break
        if self._verbose:
            print("Data source types successfully set.")
            print(f"Electron beam source type is {self.ebeam_type}")
            print(f"Gas detector source type is {self.gasdetector_type}")
        pass       

    def _set_n_pulses(self, n_pulses):
        """
        """
        self._n_pulses = n_pulses


    ##### Get data #####

    def get_data(self, evt):
        """
        Parameters
        ----------
        evt : psana.Event

        Returns
        -------
        (ebeam, gasdetector, frame) : (
            psana.Bld.BldDataEBeamVX or None,
            psana.Bld.BldDataFEEGasDetEnergyV1 or None,
            psana.Camera.FrameV1 or None,
        )
        """
        ebeam = self._get_ebeam(evt)
        gasdetector = self._get_gasdetector(evt)
        frame = self._get_frame(evt)
        return ebeam, gasdetector, frame

    def _get_frame(self,evt):
        return evt.get(self.xtcav_type, self.xtcav_camera)
    
    # Note that ._get_ebeam and ._get_gasdetector are
    # defined in ._set_data_source_types_and_getters

    def _get_shot_to_shot_parameters(self, evt, ebeam, gasdetector):
        """
        Parameters
        ----------
        evt : psana.Event
        ebeam : psana.Bld.BldDataEBeamV(X)
            ebeam object for an event
        gasdetector : psana.Bld.BldDataFEEGasDetEnergy(V1)
            gas detector object for an event

        Returns
        -------
        2-tuple : (dict, bool)
            Output dictionary has keys
                - ebeamcharge
                - dumpecharge (in C)
                - xtcavrfamp
                - xtcavrfphase
                - xrayenergy (in J)
            Bool indicates success (True) or failure (False) to retrieve data
        """
        ok=1

        # Get e-beam data
        if ebeam:    
            ebeamcharge=ebeam.ebeamCharge()
            xtcavrfamp=ebeam.ebeamXTCAVAmpl()
            xtcavrfphase=ebeam.ebeamXTCAVPhase()
            if self._patch_shot_metadata:
                xtcavrfamp = self._global_calibrations['rfampcalib']
                xtcavrfphase = self._global_calibrations['rfphasecalib']
            dumpecharge=ebeam.ebeamDumpCharge() * ECHARGE  # in C
        else:
            warnings.warn_explicit('No ebeamv info',UserWarning,'XTCAV',0)
            ok=0
            ebeamcharge=None
            xtcavrfamp=None
            xtcavrfphase=None
            energydetector=None
            dumpecharge=None

        # Get gas detector data
        if gasdetector:
            energydetector=(gasdetector.f_11_ENRC()+gasdetector.f_12_ENRC())/2    
        else:
            warnings.warn_explicit('No gas detector info',UserWarning,'XTCAV',0)
            ok=0
            energydetector=None
        energy=1e-3*energydetector  # In J

        # Event ID info
        iden = evt.get(psana.EventId)
        time = iden.time()
        sec = time[0]
        nsec = time[1]
        unixtime = int((sec<<32)|nsec)
        fiducial = iden.fiducials()

        # Overwrites
        if 'ebeamcharge' in self._shotByShot_calibration_overwrites.keys():
            ebeamcharge = self.global_shotByShot_overwrites['ebeamcharge']
        if 'dumpecharge' in self._shotByShot_calibration_overwrites.keys():
            dumpecharge = self.global_shotByShot_overwrites['dumpecharge']
        if 'xtcavrfamp' in self._shotByShot_calibration_overwrites.keys():
            xtcavrfamp = self.global_shotByShot_overwrites['xtcavrfamp']
        if 'xtcavrfphase' in self._shotByShot_calibration_overwrites.keys():
            xtcavrfphase = self.global_shotByShot_overwrites['xtcavrfphase']
        if 'xrayenergy' in self._shotByShot_calibration_overwrites.keys():
            energy = self.global_shotByShot_overwrites['xrayenergy']

        # Return
        shotToShot={
            'ebeamcharge':ebeamcharge,      # ebeamcharge
            'dumpecharge':dumpecharge,      # dumpecharge in C
            'xtcavrfamp': xtcavrfamp,       # RF amplitude
            'xtcavrfphase':xtcavrfphase,    # RF phase
            'xrayenergy':energy,            # Xrays energy in J
            'unixtime':unixtime,            # unix system time
            'fiducial':fiducial,            # event fiducial
            }        
        return shotToShot,ok


    ##### Indexing Events #####

    def _setup_indexing(self):
        """ Setup event indexing containers 
        """
        self.data_is_indexed = False
        
        # setup containers
        self._times_ebeam = []
        self._times_gasdetector = []
        self._times_frames = []
        self._times_ebeamAndGas = []
        self._times_ebeamAndFrame = []
        self._times_gasAndFrame = []
        self._times_ebeam_only = []
        self._times_gas_only = []
        self._times_frame_only = []
        self._times_ok = []
        self._times_good = []
        self._times_none = []
    
    def _preindex_shots(self):
        """ Loop through and index (which shots received data from which cameras
        i.e. ebeam detector, gas detector, & XTCAV camera) all events during instantiation
        """
        # setup containers
        self._setup_indexing()
        
        # loop
        for i,t in enumerate(self._progress(self._times,desc="Indexing shots...")):
            evt = self._run.event(t)
            self._index_single_shot(i,evt)

        # done
        if self._verbose:
            print(f"Out of {self.N_shots} total shots...")
            print()
            print(f"...found {len(self._times_ebeam)} shots with ebeam data")
            print(f"...found {len(self._times_gasdetector)} shots with gasdetector data")
            print(f"...found {len(self._times_frames)} shots with frame data")
            print()
            print(f"...found {len(self._times_ok)} shots with all 3 data sources")
            print(f"...found {len(self._times_ebeamAndGas)} shots with ebeam+gas+NOT(frame) data")
            print(f"...found {len(self._times_ebeamAndFrame)} shots with ebeam+frame+NOT(gas) data")
            print(f"...found {len(self._times_gasAndFrame)} shots with gas+frame+NOT(ebeam) data")
            print(f"...found {len(self._times_ebeam_only)} shots with ebeam data ONLY")
            print(f"...found {len(self._times_gas_only)} shots with gas data ONLY")
            print(f"...found {len(self._times_frame_only)} shots with frame data ONLY")
            print(f"...found {len(self._times_none)} shots with NO data")
            print()

        self._data_is_indexed = True
        pass

    def _index_single_shot(self, i, evt, returnvals=False):
        """ Index a single event.

        Parameters
        ----------
        i : int
            The index of the event in self._times
        evt : psana.Event
        """
        ebeam, gasdetector, frame = self.get_data(evt)
        # set individual detector indices
        if ebeam:
            self._times_ebeam.append(i)
        if gasdetector:
            self._times_gasdetector.append(i)
        if frame:
            self._times_frames.append(i)
        # set joint detector indices
        if ebeam and gasdetector and frame:
            self._times_ok.append(i)
        elif ebeam and gasdetector and not frame:
            self._times_ebeamAndGas.append(i)
        elif ebeam and not gasdetector and frame:
            self._times_ebeamAndFrame.append(i)
        elif not ebeam and gasdetector and frame:
            self._times_gasAndFrame.append(i)
        elif ebeam and not gasdetector and not frame:
            self._times_ebeam_only.append(i)
        elif not ebeam and gasdetector and not frame:
            self._times_gas_only.append(i)
        elif not ebeam and not gasdetector and frame:
            self._times_frame_only.append(i)
        else:
            self._times_none.append(i)
        if returnvals:
            return ebeam, gasdetector, frame

    ##### Iterators #####

    def reset_shot_iterator(self):
        """ reset the shot iterator index
        """
        if self._data_is_indexed:
            if self._iteration_type=='all':
                self.shot_iterator = self._setup_shot_iterator()
            elif self._iteration_type=='ok':
                self.shot_iterator = self._setup_ok_shot_iterator()
            elif self._iteration_type=='frames':
                self.shot_iterator = self._setup_frame_shot_iterator()
            elif self._iteration_type=='good':
                self.shot_iterator = self._setup_good_shot_iterator()
            else:
                raise Exception(f"Iteration type {self._iteration_type} is not supported. Must be in ('all','ok','frames','good')")
        else:
            if self._iteration_type=='all':
                self.shot_iterator = self._setup_on_the_fly_shot_iterator()
            elif self._iteration_type=='ok':
                self.shot_iterator = self._setup_ok_on_the_fly_shot_iterator()
            elif self._iteration_type=='frames':
                self.shot_iterator = self._setup_frame_on_the_fly_shot_iterator()
            elif self._iteration_type=='good':
                warnings.warn("'Good' shots have not yet been determined during initial pass; using 'ok' shot iterator.")
                self.shot_iterator = self._setup_ok_on_the_fly_shot_iterator()
            else:
                raise Exception(f"Iteration type {self._iteration_type} is not supported. Must be in ('all','ok','frames','good')")
        pass

    def _set_iteration_type(self, iteration):
        assert(iteration in ('all','ok','frames','good')), f"Iteration type {iteration} is not supported. Must be in ('all','ok','frames','good')"
        self._iteration_type = iteration
        pass

    def reset_iteration_type(self, iteration):
        """ Update the shot iterator type, then reset the iterator.

        Parameters
        iteration : string in ('all', 'ok', 'frames', 'good'):
        """
        self._set_iteration_type(iteration)
        self.reset_shot_iterator()
        pass

    # Pre-indexed iterators
    def _setup_shot_iterator(self):
        """ Create iterator looping over all shots.
        """
        idx = 0
        while idx < self.N_shots:
            yield idx,idx,self.get_event(idx)
            idx += 1
        self.reset_shot_iterator()
        print(f"All {self.N_shots} shots have been traversed.")
        print("Resetting the shot iterator.")
        pass
    def _setup_ok_shot_iterator(self):
        """ Create iterator looping over all shots with XTCAV camera, ebeam,
        and gas detector data present.
        """
        idx = 0
        while idx < len(self._times_ok):
            idx_times = self._times_ok[idx]
            yield idx,idx_times,self.get_event(idx_times)
            idx += 1
        self.reset_shot_iterator()
        print(f"All {self.N_shots} shots have been traversed.")
        print("Resetting the shot iterator.")
        pass
    def _setup_frame_shot_iterator(self):
        """ Create iterator looping over all shots containing XTCAV camera data.
        """
        idx = 0
        while idx < len(self._times_frames):
            idx_times = self._times_frames[idx]
            yield idx,idx_times,self.get_event(idx_times)
            idx += 1
        self.reset_shot_iterator()
        print(f"All {self.N_shots} shots have been traversed.")
        print("Resetting the shot iterator.")
        pass
    def _setup_good_shot_iterator(self):
        """ Create iterator looping over all usable (all stats calculable) shots.
        """
        idx = 0
        while idx < len(self._times_good):
            idx_times = self._times_good[idx]
            yield idx,idx_times,self.get_event(idx_times)
            idx += 1
        self.reset_shot_iterator()
        print(f"All {self.N_shots} shots have been traversed.")
        print("Resetting the shot iterator.")
        pass

    # On-the-fly iterators
    def _setup_on_the_fly_shot_iterator(self):
        """ Create iterator looping over all shots, indexing shots on-the-fly.
        """
        self._setup_indexing()
        idx = 0
        while idx < self.N_shots:
            event = self.get_event(idx)
            self._index_single_shot(idx,event)
            yield idx,idx,event
            idx += 1
        self._data_is_indexed = True
        self.reset_shot_iterator()
        print(f"All {self.N_shots} shots have been traversed.")
        print("Resetting the shot iterator.")
        pass
    def _setup_ok_on_the_fly_shot_iterator(self):
        """ Create iterator looping over all shots, indexing shots on-the-fly.
        """
        self._setup_indexing()
        idx = 0
        jdx = 0
        while idx < self.N_shots:
            event = self.get_event(idx)
            ebeam, gasdetector, frame = self._index_single_shot(idx,event,returnvals=True)
            if ebeam and gasdetector and frame:
                yield jdx,idx,event
                jdx += 1
            else:
                pass
            idx += 1
        self._data_is_indexed = True
        self.reset_shot_iterator()
        print(f"All {self.N_shots} shots have been traversed.")
        print("Resetting the shot iterator.")
        pass
    def _setup_frame_on_the_fly_shot_iterator(self):
        """ Create iterator looping over all shots, indexing shots on-the-fly.
        """
        self._setup_indexing()
        idx = 0
        jdx = 0
        while idx < self.N_shots:
            event = self.get_event(idx)
            ebeam, gasdetector, frame = self._index_single_shot(idx,event,returnvals=True)
            if frame:
                yield jdx,idx,event
                jdx += 1
            else:
                pass
            idx += 1
        self._data_is_indexed = True
        self.reset_shot_iterator()
        print(f"All {self.N_shots} shots have been traversed.")
        print("Resetting the shot iterator.")
        pass

    # User facing pre-indexed iterator generators
    def generate_complete_shots_iterator(self):
        """ Create an iterator looping over all shots with all data sources present
        """
        assert(self._data_is_indexed), "This iterator can't be used unless data is indexed!"
        idx = 0
        while idx < len(self._times_ok):
            idx_times = self._times_ok[idx]
            yield idx,idx_times,self.get_event(idx_times)
            idx += 1
    def generate_ebeam_shots_iterator(self):
        """ Create an iterator looping over all shots with ebeam data present
        """
        assert(self._data_is_indexed), "This iterator can't be used unless data is indexed!"
        idx = 0
        while idx < len(self._times_ebeam):
            idx_times = self._times_ebeam[idx]
            yield idx,idx_times,self.get_event(idx_times)
            idx += 1
    def generate_gasdetector_shots_iterator(self):
        """ Create an iterator looping over all shots with gasdetector data present
        """
        assert(self._data_is_indexed), "This iterator can't be used unless data is indexed!"
        idx = 0
        while idx < len(self._times_gasdetector):
            idx_times = self._times_gasdetector[idx]
            yield idx,idx_times,self.get_event(idx_times)
            idx += 1
    def generate_frame_shots_iterator(self):
        """ Create an iterator looping over all shots with xtcav camera data present
        """
        assert(self._data_is_indexed), "This iterator can't be used unless data is indexed!"
        idx = 0
        while idx < len(self._times_frames):
            idx_times = self._times_frames[idx]
            yield idx,idx_times,self.get_event(idx_times)
            idx += 1

    ##### Event Setter & Getter #####
        
    def set_current_event(self,evt):
        """
        Sets the current psana event.
        Once this is run it is possible to query for results such as X-Ray power, or pulse delay.

        Parameters
        ----------
        evt : psana.Event

        Returns
        -------
        (bool,bool,bool)
            Flags indicating if (ebeam data, gas detector data, frame data) are present
        """
        # Because the calculations to get the reconstruction will not be done until the
        # information itself is requested, calls to this method should be quite fast.

        # Get data
        ebeam, gasdetector, frame = self.get_data(evt)

        # Set data
        self._currentevent=evt
        try:
            self._rawimage=frame.data16().astype(np.float64)
            if self._test_xy: self._rawimage = self._rawimage[:,:-1]
        except AttributeError:
            self._rawimage=None
        self._ebeam=ebeam
        self._gasdetector=gasdetector
        #self._currenteventavailable=True

        # Return
        return (self._ebeam is not None, self._gasdetector is not None, self._rawimage is not None)

    def get_event(self,idx):
        """
        Gets the event at time index `idx`.

        Parameters
        ----------
        idx : int

        Returns
        -------
        psana.Event
        """
        return self._run.event(self._times[idx])



    ##### Data Access Properties #####

    @property
    def N_shots(self):
        """ Number of shots """
        return len(self._times)
    @property
    def ebeam(self):
        return self._ebeam
    @property
    def gasdetector(self):
        return self._gasdetector
    @property
    def frame(self):
        return self._rawimage
    @property
    def frameshape(self):
        return self._xtcav_frame_shape
    @property
    def nx(self):
        return self._xtcav_frame_shape[1]
    @property
    def ny(self):
        return self._xtcav_frame_shape[0]
    @property
    def x(self):
        return self._xtcav_frame_x
    @property
    def y(self):
        return self._xtcav_frame_y
    @property
    def xx(self):
        return self._xtcav_frame_xx
    @property
    def yy(self):
        return self._xtcav_frame_yy
    @property
    def run(self):
        return self._run
    @property
    def times(self):
        return self._times
    @property
    def dark_reference(self):
        return self._dark_reference
    @property
    def n_pulses(self):
        return self._n_pulses


    ##### Preprocess Frames #####

    def set_darkreference(self, darkref):
        """
        Parameters
        ----------
        darkref : 2d array or XTCAVDarkReference
        """
        try:
            ans = darkref.dark_reference
            if ans is None:
                raise Exception("An XTCAVDarkReference object was passed but its .dark_reference has not been calculated; try running .compute()!")
        except AttributeError:
            assert(isinstance(darkref,np.ndarray)), f"darkref must be an XTCAVDarkReference instance or a numpy array, not type {type(darkref)}"
            ans = darkref
        self._dark_reference = ans
        pass

    def get_frame_bksb(self, frame):
        """
        Returns a frame after subtracting the dark reference

        Parameters
        ----------
        frame : 2d array

        Returns
        -------
        2d array
        """
        try:
            ans = frame - self.dark_reference
        except AttributeError:
            raise Exception("Dark reference has not been set - try running .set_darkreference()!")
        ans = np.maximum(ans,0)
        return ans

    def get_frame_bksb_current(self):
        """ Returns the current frame after subtracting the dark reference
        """
        return self.get_frame_bksb(self.frame)

    def get_frame_denoised(self, frame, nstd=None, thresh_mode=None, close=None,
        noise_x=None, noise_y=None, noise_x_end=None, noise_y_end=None,
        medfilt_size=None, normalize=None, returnall=False):
        """
        Returns a frame after denoising.  

        First sets pixels in a binary masked region to zero, where the mask is
        determined first by thresholding then by applying a binary closing.  The
        threshold value is given by AVE + N*STD, N is the `nstd` parameter, and
        AVE and STD are the average (median or mean) and standard deviation,
        respectively, of a region of the image believed to be comprised entirely
        of noise specified by the `noise_*` parameters. Then applies a median
        filter. Finally, normalizes the image, and returns.

        Parameters
        ----------
        frame : 2d array
            The image
        nstd : number
            Threshold is set to this number of standard deviations above the
            average
        thresh_mode : 'median' or 'mean'
            Sets how the average is determined
        close : int
            After thresholding, perform a binary closing with this number of
            iterations
        noise_x : int
            The noise statistics region's extent in x
        noise_y : int
            The noise statistics region's extent in y
        noise_x_end : bool
            Toggles whether the noise statistics regions is taken from the front
            (False) or back (True) of the image array in x
        noise_y_end : bool
            Toggles whether the noise statistics regions is taken from the front
            (False) or back (True) of the image array in x
        medfilt_size : int
            The median filter size or footprint. If set to 1 (default), no median
            filter is applied
        normalize : bool
            Toggles normalizing total image intensity to 1
        returnall : bool
            Toggles returning only the output frame vs. returning the mask and
            noise subframe mask as well

        Returns
        -------
        2d array
        """
        # update params
        p = {}
        if nstd is not None: p['nstd'] = nstd
        if thresh_mode is not None: p['thresh_mode'] = thresh_mode
        if close is not None: p['close'] = close
        if noise_x is not None: p['noise_x'] = noise_x
        if noise_y is not None: p['noise_y'] = noise_y
        if noise_x_end is not None: p['noise_x_end'] = noise_x_end
        if noise_y_end is not None: p['noise_y_end'] = noise_y_end
        if medfilt_size is not None: p['medfilt_size'] = medfilt_size
        if normalize is not None: p['normalize'] = normalize
        self._update_denoise_params(**p)
        params = self._denoise_params
        nstd = params['nstd']
        thresh_mode = params['thresh_mode']
        close = params['close']
        noise_x = params['noise_x']
        noise_y = params['noise_y']
        noise_x_end = params['noise_x_end']
        noise_y_end = params['noise_y_end']
        medfilt_size = params['medfilt_size']
        normalize = params['normalize']
        # get noise statistics
        roi_noise_x = (-noise_x,frame.shape[1]) if noise_x_end else (0,noise_x)
        roi_noise_y = (-noise_y,frame.shape[0]) if noise_y_end else (0,noise_y)
        subframe = frame[roi_noise_y[0]:roi_noise_y[1],roi_noise_x[0]:roi_noise_x[1]]
        assert(thresh_mode in ('median', 'mean')), f"thresh_mode must be 'median' or 'mean', not {thresh_mode}."
        ave = np.median(subframe) if thresh_mode=='median' else np.mean(subframe)
        # get threshold & mask
        thresh = ave + nstd*np.std(subframe)
        mask = frame<thresh
        # binary closing
        mask = binary_closing(mask, iterations=close)
        # fix edges
        if close>0:
            c=close
            mask[:c,:]=1
            mask[-c:,:]=1
            mask[:,:c]=1
            mask[:,-c:]=1
        # apply mask, median filter, and normalization
        ans = frame*np.logical_not(mask)
        ans = median_filter(ans, medfilt_size)
        if normalize:
            ans /= np.sum(ans)
        # return
        if returnall:
            subframe_mask = np.zeros(frame.shape,dtype=bool)
            subframe_mask[roi_noise_y[0]:roi_noise_y[1],roi_noise_x[0]:roi_noise_x[1]] = 1
            return ans, mask, subframe_mask
        else:
            return ans

    def get_frame_denoised_current(self, bksb=None, nstd=None, thresh_mode=None, close=None,
        noise_x=None, noise_y=None, noise_x_end=None, noise_y_end=None,
        medfilt_size=None, normalize=None, returnall=False):
        """
        Returns a frame after denoising. See .get_frame_denoised(). 

        Parameters
        ----------
        bksb : bool
            Toggles applying dark reference subtraction before denoising
        """
        # update params
        p = {}
        if bksb is not None: p['bksb'] = bksb
        if nstd is not None: p['nstd'] = nstd
        if thresh_mode is not None: p['thresh_mode'] = thresh_mode
        if close is not None: p['close'] = close
        if noise_x is not None: p['noise_x'] = noise_x
        if noise_y is not None: p['noise_y'] = noise_y
        if noise_x_end is not None: p['noise_x_end'] = noise_x_end
        if noise_y_end is not None: p['noise_y_end'] = noise_y_end
        if medfilt_size is not None: p['medfilt_size'] = medfilt_size
        if normalize is not None: p['normalize'] = normalize
        self._update_denoise_params(**p)
        params = self._denoise_params
        bksb = params['bksb']
        nstd = params['nstd']
        thresh_mode = params['thresh_mode']
        close = params['close']
        noise_x = params['noise_x']
        noise_y = params['noise_y']
        noise_x_end = params['noise_x_end']
        noise_y_end = params['noise_y_end']
        medfilt_size = params['medfilt_size']
        normalize = params['normalize']
        # background subtraction
        if bksb:
            frame = self.get_frame_bksb_current()
        else:
            frame = self.frame
        # get and return the answer
        return self.get_frame_denoised(
            frame=frame,
            nstd=nstd,
            thresh_mode=thresh_mode,
            close=close,
            noise_x=noise_x,
            noise_y=noise_y,
            noise_x_end=noise_x_end,
            noise_y_end=noise_y_end,
            medfilt_size=medfilt_size,
            normalize=normalize,
            returnall=returnall
        )

    def split_frame(self, frame, n_pulses=None, method='connected'):
        """
        For multi-pulse shot modes. Split an XTCAV image frame into subframes, each
        containing the data from a single pulse.

        Parameters
        ----------
        frame : 2d array
            The image
        n_pulses : int or None
            The number of pulses.  Should be 1 or 2 or None. If None uses the currently
            set value (defaults to 1).
        method : string in ('connected','autothreshold','kmeans','spectral','dbscan')
            The image processing method used to split the images.  Currently
            (6/2026) only 'connected' is implemented and corresponds to what's
            called the "island method" in the original xtcav code

        Returns
        -------
        3d array of shape (n_pulses, nx, ny)
        """
        # Get number of pulses
        if n_pulses is not None:
            self._set_n_pulses(n_pulses)
        else:
            n_pulses = self.n_pulses
        # For one pulse, return it
        if n_pulses==1:
            return frame[np.newaxis,:,:]
        # For two pulses, split the frame
        elif n_pulses==2:
            assert(method in ('connected','authothreshold','kmeans','spectral','dbscan')), f"Unrecognized frame splitting method {method}."
            if method == 'connected':
                return self.split_frame_connected(frame)
            elif method == 'autothreshold':
                return self.split_frame_autothreshold(frame)
            elif method == 'kmeans':
                return self.split_frame_kmeans_cluster(frame)
            elif method == 'spectral':
                return self.split_frame_spectral_cluster(frame)
            elif method == 'dbscan':
                return self.split_frame_dbscan_cluster(frame)
            else:
                raise Exception(f"Unrecognized frame splitting method {method}.")
        # Only 1 or 2 pulses are currently supported
        else:
            raise Exception(f"Only 1 or 2 pulse modes are currently supported. {n_pulses} pulses are not supported.")




    ##### Data Processing #####

    def get_pulse_statistics(self, frame):
        """
        Get statistics for a pulse.

        Collected statistics include:
            - the total image frame intensity
            - the total intensity profile projected along the x (time) axis
            - the center of mass in x (time)
            - the standard deviation in x (time)
            - the FWHM of the x-projected intensity profile
            - the total intensity profile projected along the y (energy) axis
            - the center of mass in y (energy)
            - the standard deviation in y (energy)
            - the FWHM of the y-projected intensity profile
            - the y-projected COM for each x-slice
            - the y-projected STD for each x-slice
            - a flag indicating success or failure of these calculations

        Parameters
        ----------
        frame : 2d array
            An image. Should be a single-pulse image, i.e. for multi-pulse data, this
            should be the frame after splitting has been performed.

        Returns
        -------
        dictionary with fields
            - 'imFrac'
            - 'xProfile'
            - 'xCOM'
            - 'xRMS'
            - 'xFWHM',
            - 'yProfile'
            - 'yCOM'
            - 'yRMS'
            - 'yFWHM'
            - 'yCOMslice'
            - 'yRMSslice'
            - 'success'
        """
        # Total image intensity; should be 1 for single-pulse and <1 for multi-pulse data
        imFrac = np.sum(frame)

        # Is there a pulse to look at?
        if imFrac==0:
            xProfile=-1
            xCOM=-1
            xRMS=-1
            xFWHM=-1
            yProfile=-1
            yCOM=-1
            yRMS=-1
            yFWHM=-1
            yCOMslice=-1
            yRMSslice=-1
            success = False
        
        else:

            # Get stats in x (time)
            xProfile=np.sum(frame,0)                                    # Profile projected onto the x axis
            xCOM=np.dot(xProfile,np.transpose(self.x))/imFrac           # X position of the center of mass
            xRMS= np.sqrt(np.dot((self.x-xCOM)**2,xProfile)/imFrac)     # Standard deviation of the values in x
            ind=np.where(xProfile >= np.amax(xProfile)/2)[0]
            xFWHM=np.abs(ind[-1]-ind[0]+1)                              # FWHM of the X profile

            # Get stats in y (energy)
            yProfile=np.sum(frame,1)                                    # Profile projected onto the y axis
            yCOM=np.dot(yProfile,self.y)/imFrac                         # Y position of the center of mass
            yRMS= np.sqrt(np.dot((self.y-yCOM)**2,yProfile)/imFrac)     # Standard deviation of the values in y
            ind=np.where(yProfile >= np.amax(yProfile)/2)
            yFWHM=np.abs(ind[-1]-ind[0])                                # FWHM of the Y profile

            yCOMslice=self.divideNoWarn(np.dot(np.transpose(frame),self.y),xProfile,yCOM)    # Y position of the center of mass for each slice in x
            distances=np.outer(np.ones(yCOMslice.shape[0]),self.y)-np.outer(yCOMslice,np.ones(frame.shape[0]))    #For each point of the image, the distance to the y center of mass of the corresponding slice
            yRMSslice= self.divideNoWarn(np.sum(np.transpose(frame)*((distances)**2),1),xProfile,0)         #Width of the distribution of the points for each slice around the y center of masses                  
            yRMSslice = np.sqrt(yRMSslice)

            success = True

        # Store the answer and return
        imageStats={
            'imFrac':imFrac,
            'xProfile':xProfile,
            'xCOM':xCOM,
            'xRMS':xRMS,
            'xFWHM':xFWHM,
            'yProfile':yProfile,
            'yCOM':yCOM,
            'yRMS':yRMS,
            'yFWHM':yFWHM,
            'yCOMslice':yCOMslice,
            'yRMSslice':yRMSslice,
            'success':success
        }
        return imageStats

    def calculate_physical_units(self, center, shotToShot):
        """
        Calculate physical units for the x (time) and y (energy) axes.

        Parameters
        ----------
        center : len-2 array
            Coordinates of the center of mass
        shotToShot : dict
            Properties of the specific shot
        
        Returns
        -------
        2-tuple : (dict, bool)
            (physical units, success flag)
        """
        # success flag
        ok=1

        # get global calibration values
        umperpix=self._global_calibrations['umperpix']
        #dumpe=self._global_calibrations['dumpe']  # this doesn't exist in L10376...
        dumpdisp=self._global_calibrations['dumpdisp']    
        rfampcalib=self._global_calibrations['rfampcalib']
        rfphasecalib=self._global_calibrations['rfphasecalib']    
        strstrength=self._global_calibrations['strstrength']
        
        # get shot-to-shot calibration values
        rfamp=shotToShot['xtcavrfamp']
        rfphase=shotToShot['xtcavrfphase']
        #dumpe=shotToShot['dumpecharge']  # ?????

        # get pixel conversions
        yMeVPerPix = umperpix*dumpe/dumpdisp*1e-3                   # y pixel size (MeV)
        xfsPerPix = -umperpix*rfampcalib/(0.3*strstrength*rfamp)    # x pixel size (fs)

        # time axis phase handling - (1) confirm ~linearity, (2) find positive dir
        cosphasediff=math.cos((rfphasecalib-rfphase)*math.pi/180)
        # (1) If the cosine of phase was too close to 0, raise warning and return failure
        if np.abs(cosphasediff)<0.5:
            warnings.warn_explicit('The phase of the bunch with the RF field is far from 0 or 180 degrees',UserWarning,'XTCAV',0)
            ok=0
        # (2) Check direction of the time axis
        signflip = np.sign(cosphasediff)
        xfsPerPix = signflip*xfsPerPix
        
        # calculate x- and y- axis vectors with physical units
        xfs=xfsPerPix*(self.x-center[0])            # in fs
        yMeV=yMeVPerPix*(self.y-center[1])          # in MeV

        # prepare answer and return
        physicalUnits={
            'yMeVPerPix':yMeVPerPix,
            'xfsPerPix':xfsPerPix,
            'xfs':xfs,
            'yMeV':yMeV
            }
        return physicalUnits,ok

    def get_physical_units_HARDCODED(self, center, shotToShot):
        """
        Retrieve physical units which have been hardcoded.

        This method is a patch / temporary workaround for the problem of missing metadata
        which is preventing proper calibration of the XTCAV camera's physical units.
        Normally physical units should be found with .calculate_physical_units; this
        method is a stand-in, and therefore accepts identical inputs.

        Parameters
        ----------
        center : len-2 array
            Coordinates of the center of mass
        shotToShot : dict
            Properties of the specific shot
        
        Returns
        -------
        2-tuple : (dict, bool)
            (physical units, success flag)
        """
        ## success flag
        #ok=1

        ## get global calibration values
        #umperpix=self._global_calibrations['umperpix']
        #dumpe=self._global_calibrations['dumpe']
        #dumpdisp=self._global_calibrations['dumpdisp']    
        #rfampcalib=self._global_calibrations['rfampcalib']
        #rfphasecalib=self._global_calibrations['rfphasecalib']    
        #strstrength=self._global_calibrations['strstrength']
        
        ## get shot-to-shot calibration values
        #rfamp=shotToShot['xtcavrfamp']
        #rfphase=shotToShot['xtcavrfphase']

        ## get pixel conversions
        #yMeVPerPix = umperpix*dumpe/dumpdisp*1e-3                   # y pixel size (MeV)
        #xfsPerPix = -umperpix*rfampcalib/(0.3*strstrength*rfamp)    # x pixel size (fs)

        ## time axis phase handling - (1) confirm ~linearity, (2) find positive dir
        #cosphasediff=math.cos((rfphasecalib-rfphase)*math.pi/180)
        ## (1) If the cosine of phase was too close to 0, raise warning and return failure
        #if np.abs(cosphasediff)<0.5:
        #    warnings.warn_explicit('The phase of the bunch with the RF field is far from 0 or 180 degrees',UserWarning,'XTCAV',0)
        #    ok=0
        ## (2) Flip time axis direction if needed
        #signflip = np.sign(cosphasediff)
        #xfsPerPix = signflip*xfsPerPix;    
        
        ## calculate x- and y- axis vectors with physical units
        #xfs=xfsPerPix*(self.x-center[0])            # in fs
        #yMeV=yMeVPerPix*(self.y-center[1])          # in MeV
        

        # Raise a warning: these have not been correctly calibrated yet
        warnings.warn("Hardcoded calibrations are not yet calculated - returning uncalibrated axes!")
        ok = 1

        ### Set physical units
        yMeVPerPix = 1.0     # y pixel size (MeV)
        xfsPerPix = 1.0      # x pixel size (fs)

        ## calculate x- and y- axis vectors with physical units
        xfs=xfsPerPix*(self.x-center[0])            # in fs
        yMeV=yMeVPerPix*(self.y-center[1])          # in MeV
        
        # prepare answer and return
        physicalUnits={
            'yMeVPerPix':yMeVPerPix,
            'xfsPerPix':xfsPerPix,
            'xfs':xfs,
            'yMeV':yMeV
            }
        return physicalUnits,ok


    def get_shot_by_shot_statistics(self,n_pulses=None,_n_shots_max=None):
        """
        """
        # Get number of pulses
        if n_pulses is not None:
            self._set_n_pulses(n_pulses)
        else:
            n_pulses = self.n_pulses

        # Containers
        self._pulse_statistics=[[] for i in range(n_pulses)]
        self._shot_to_shot_params=[]
        self._physical_units=[]
        self._times_good=[]

        # Prep for loop
        total = self.N_shots if _n_shots_max is None else _n_shots_max
        self.reset_shot_iterator()
        progress_bar = tqdm(
            desc="Calculating shot-by-shot statistics...",
            total=total
        )

        # Loop
        for idx,(i,j,evt) in enumerate(self.shot_iterator):
            # Get event & data
            self.set_current_event(evt)
            ebeam, gasdetector, frame = self.get_data(evt)
            shotToShot,ok = self._get_shot_to_shot_parameters(evt,ebeam,gasdetector)
            if not ok:
                continue

            # Check if frame is saturated
            if np.max(self.frame)>self._camera_saturation_value:
                continue

            # Check if ROI is too close to frame edge
            com = self.get_com_clean(self.frame)
            fov = self._vis_params['fov']
            if com[0]-fov[0]/2 < -fov[0]*0.1 or \
                com[0]+fov[0]/2 >= 1023+fov[0]*0.1 or \
                com[1]-fov[1]/2 < -fov[1]*0.1 or \
                com[1]+fov[1]/2 >= 1023+fov[1]*0.1:
                #warnings.warn(f"Shot {j} is outside the XTCAV camera frame")
                continue

            # Prepare pulse images - denoise & split
            im = self.get_frame_denoised_current()
            ims,ok = self.split_frame(im,n_pulses)
            if not ok:
                continue
            
            # Get statistics
            imageStats = []
            for jdx in range(n_pulses):
                imageStats.append(self.get_pulse_statistics(ims[jdx]))
            
            # Get physical units
            # TODO fix with proper unit calibration once we have correct metadata
            # TODO fix to handle double shots correctly
            #physical_units, ok = xtpr.calculate_physical_units(   # this line should be used once we have the correct metadata
            physical_units, ok = self.get_physical_units_HARDCODED(
                center = np.array([imageStats[0]['xCOM'],imageStats[0]['yCOM']]),
                shotToShot = shotToShot
            )
            if not ok:
                continue

            # If the time step is negative, mirror the x-axis
            if physical_units['xfsPerPix']<0:
                physical_units['xfs'] = physical_units['xfs'][::-1]
                for jdx in range(n_pulses):
                    imageStats[jdx]['xProfile'] = imageStats[jdx]['xProfile'][::-1]
                    imageStats[jdx]['yCOMslice'] = imageStats[jdx]['yCOMslice'][::-1]
                    imageStats[jdx]['yRMSslice'] = imageStats[jdx]['yRMSslice'][::-1]

            # Store outputs
            if ok:
                for jdx in range(n_pulses):
                    self._pulse_statistics[jdx].append(imageStats[jdx])
                self._shot_to_shot_params.append(shotToShot)
                self._physical_units.append(physical_units)
                self._times_good.append(j)
            
            # Update iterator
            progress_bar.n = j+1
            progress_bar.refresh()

            # TODO - remove
            if _n_shots_max is not None:
                if idx>=_n_shots_max-1:
                    break
        progress_bar.close()
        print(f"Done. Calculated statistics for {len(self._times_good)} 'good' shots.")
        print("Setting shot iterator to traverse good shots only.")
        self.reset_iteration_type('good')
        pass



    # TODO





    ##### Processing Utilities #####

    @staticmethod
    def get_com(ar):
        """
        Return the 2D center of mass

        Parameters
        ----------
        ar : 2d array
            The object

        Returns
        -------
        [x,y]
        """
        s = ar.shape
        xx,yy = np.meshgrid(np.arange(s[1]),np.arange(s[0]))
        m = np.sum(ar)
        x = np.sum(xx*ar)/m
        y = np.sum(yy*ar)/m
        return np.array([y,x])

    @staticmethod
    def get_frame_mask(frame, thresh=None, thresh_mode='median', std=1, close=1):
        """
        Get a mask for an xtcav camera acquisition.

        Pixel values below a threshold are masked, then a binary closing operation is performed.
        The threshold can be set manually with the `thresh` parameter, or will be automatically
        set to AVE + N*STD, where AVE is the median or mean (set by `thresh_mode`), STD is the
        standard deviation, and N a number (set by `std`).

        Parameters
        ----------
        frame : 2d array
            The image
        thresh : number or None
            The threshold value.  If None, set automatically
        thresh_mode : 'median' or 'mean'
            Sets how the average is determined
        std : number
            Threshold is set to this number of standard deviations above the average
        close : int
            After thresholding, perform a binary closing with this number of iterations

        Returns
        -------
        2d boolean array
        """
        # set the threshold
        if thresh is None:
            assert(thresh_mode in ('median', 'mean')), f"thresh_mode must be 'median' or 'mean', not {thresh_mode}."
            # get average
            if thresh_mode == 'median':
                thresh = np.median(frame)
            else:
                thresh = np.mean(frame)
            # add std
            thresh += np.std(frame) * std
        # get mask
        mask = frame<thresh
        # remove stray pixels
        mask = binary_closing(mask, iterations=close)
        # fix edges
        if close>0:
            c=close
            mask[:c,:]=1
            mask[-c:,:]=1
            mask[:,:c]=1
            mask[:,-c:]=1
        # return
        return mask

    def get_current_frame_mask(self, thresh=None, thresh_mode='median', std=1, close=1):
        """ See `.get_frame_mask`
        """
        assert(self.frame is not None), "No XTCAV camera image data exists for the current event."
        return self.get_frame_mask(
            self.frame,
            thresh=thresh,
            thresh_mode=thresh_mode,
            std=std,
            close=close,
        )

    def get_com_clean(self, frame, thresh=None, thresh_mode='median', std=1, close=2, returnmask=False):
        """
        Return the 2D center of mass, ignoring certain masked pixel values.
        See `get_frame_mask` docstring for masking details.

        Parameters
        ----------
        frame : 2d array
            The camera image
        thresh : number or None
            See `get_frame_mask` for details
        thresh_mode : 'median' or 'mean'
            See `get_frame_mask` for details
        std : number
            See `get_frame_mask` for details
        close : int
            See `get_frame_mask` for details
        returnmask : bool
            Toggles returning the mask with the answer. If true, return value
            becomes a 2-tuple ([y,x],mask)

        Returns
        -------
        [y,x] or ([y,x],mask)
        """
        # get mask
        mask = self.get_frame_mask(
            frame,
            thresh=thresh,
            thresh_mode=thresh_mode,
            std=std,
            close=close
        )
        # compute & return
        _ar = np.copy(frame)
        _ar[mask] = 0
        ans = self.get_com(_ar)
        if returnmask:
            return ans, mask
        else:
            return ans

    def get_current_com_clean(self, thresh=None, thresh_mode='median', std=1, close=2, returnmask=False):
        """ See `.get_com_clean`
        """
        assert(self.frame is not None), "No XTCAV camera image data exists for the current event."
        return self.get_com_clean(
            self.frame,
            thresh=thresh,
            thresh_mode=thresh_mode,
            std=std,
            close=close,
            returnmask=returnmask
        )

    def split_frame_connected(self, frame, n_bunches=2):
        """
        Split a frame by connected regions.  This is called the "island method" in the original xtcav code.

        Parameters
        ----------
        frame : 2d array
            The image
        n_bunches : integer
            Number of bunches

        Returns
        -------
        2 tuple: (array, bool)
            array = 3d array of shape (2,ny,nx)
            bool = success flag
        """
        # Get a boolean image & find connected groups
        imgbool=frame>0
        groups, n_groups = label(imgbool,return_num=True);
        
        # Obtain the areas
        areas = np.zeros(n_groups,dtype=np.float64)
        for i in range(0,n_groups):    
            areas[i]=np.sum(frame*(groups==(i+1)))
 
        # Index in order of descending area
        orderareaind=np.argsort(areas)  
        orderareaind=np.flipud(orderareaind)
        if len(orderareaind)==0:
            return np.zeros((n_bunches,self.ny,self.nx)), False

        # Discard bunches with areas smaller than 1/20'th the largest area
        n_area_valid=1
        area_ref = areas[orderareaind[0]]
        for i in range(1,n_groups): 
            if areas[orderareaind[i]]<1.0/20*area_ref:
                break
            else:
                n_area_valid+=1

        # Set number of output images
        n_output=np.amin([n_bunches,n_groups,n_area_valid])
        if n_output != n_bunches:
            return np.zeros((n_bunches,self.ny,self.nx)), False
        assert(n_output==n_bunches), f"Failed to identify {n_bunches} distinct bunches with the island method; found only {n_output}."

        #Obtain the separated images
        images=[]
        for i in range(0,n_output):    
            images.append(frame*(groups==(orderareaind[i]+1)))
        
        # Order the outputs based on angular distribution
        # Find the angle of each island's CoM w.r.t the image CoM
        y0,x0=self.get_com(frame)        
        angles=np.zeros(n_output,dtype=np.float64)
        xi=np.zeros(n_output,dtype=np.float64)
        yi=np.zeros(n_output,dtype=np.float64)
        for i in range(n_output):
            yi[i],xi[i]=self.get_com(images[i])
            angles[i]=degrees(np.arctan2(yi[i]-y0,xi[i]-x0))
                    
        #If the distance of one of the islands to -180/180 angle is smaller than a certain fraction, we add an angle to make sure that nothing is close to the zero angle
        #Ordering in angles (counterclockwise from 3 oclock)
        #dist=180-abs(angles)
        #if np.amin(dist)<30.0/n_valid:
        #    angles=angles+180.0/n_valid
        #orderangleind=np.argsort(angles)  

        # Order for outputs (higher energy first)
        orderangleind=np.argsort(-yi)  

        # Assign the output
        ans=np.zeros((n_output,self.ny,self.nx))
        for i in range(n_output):
            ans[i,:,:]=images[orderangleind[i]]
        
        # Images should already be normalized at this point! Commenting this out
        #Renormalize to total area of 1
        #outimages=outimages/np.sum(outimages)
        
        # Return
        return ans, True

    @staticmethod
    def split_frame_auththreshold(frame):
        """
        """
        raise Exception("The .split_frame_authothreshold method has not yet been implemented!")
        pass

    @staticmethod
    def split_frame_kmeans_cluster(frame):
        """
        """
        raise Exception("The .split_frame_kmeans_cluster method has not yet been implemented!")
        pass

    @staticmethod
    def split_frame_spectral_cluster(frame):
        """
        """
        raise Exception("The .split_frame_spectral_cluster method has not yet been implemented!")
        pass

    @staticmethod
    def split_frame_dbscan_cluster(frame):
        """
        """
        raise Exception("The .split_frame_dbscan_cluster method has not yet been implemented!")
        pass

    @staticmethod
    def divideNoWarn(numer,denom,default):
        """ Divides numer by denom, setting invalid outputs to default
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            ratio=numer/denom
            ratio[ ~ np.isfinite(ratio)]=default  # NaN/+inf/-inf 
        return ratio



    ##### Visualization #####

    def show_frame(self, frame, vrange=None, fov=None, bins=None, hist=True, zoom=True, update_params=True, returnfig=False):
        """
        """
        # Get params
        if update_params:
            self._update_vis_params(vrange=vrange,bins=bins,fov=fov)
        vrange = vrange if vrange is not None else self._vis_params['vrange']
        bins = bins if bins is not None else self._vis_params['bins']
        fov = fov if fov is not None else self._vis_params['fov']
        # Setup figure
        if hist and zoom:
            fig,axs = plt.subplots(1,3,figsize=(15,5))
            ax1,ax2,ax3 = axs
        elif hist:
            fig,axs = plt.subplots(1,2,figsize=(10,5))
            ax1,ax3 = axs
        elif zoom:
            fig,axs = plt.subplots(1,2,figsize=(10,5))
            ax1,ax2 = axs
        else:
            fig,axs = plt.subplots(1,1,figsize=(5,5))
            ax1 = axs

        # Image
        if vrange is None:
            ax1.matshow(frame)
        else:
            ax1.matshow(frame,vmin=vrange[0],vmax=vrange[1])
        ax1.invert_yaxis()

        # Zoom-in
        if zoom:
            # get CoM
            com = self.get_com_clean(frame)
            # get FoV
            Lx,Ly = fov
            # plot zoom-in
            if vrange is None:
                ax2.matshow(frame)
            else:
                ax2.matshow(frame,vmin=vrange[0],vmax=vrange[1])
            ax2.set_xlim(self.get_plot_lims(com[1],Lx))
            ax2.set_ylim(self.get_plot_lims(com[0],Ly))
            # add zoom rectangle
            xy = (com[1]-Lx/2,com[0]-Ly/2)
            rect = Rectangle((xy),Lx,Ly,lw=2,edgecolor='r',facecolor='none')
            ax1.add_patch(rect)

        # Histogram
        if hist:
            if bins is None:
                if vrange is None:
                    ax3.hist(frame.ravel(), bins=100)
                else:
                    ax3.hist(frame.ravel(), bins=np.linspace(vrange[0],vrange[1],100))
            else:
                ax3.hist(frame.ravel(), bins=bins)
            ax3.semilogy()
        # Return
        if returnfig:
            return fig,axs
        else:
            plt.show()
            pass

    def show_current_frame(self, vrange=None, fov=None, bins=None, hist=True, zoom=True, preprocess=None, update_params=True, returnfig=False):
        """
        Parameters
        ----------
        preprocess : None or string in ('bksb', 'denoise', 'denoise_nonorm')
        """
        assert(self.frame is not None), "No XTCAV camera image data exists for the current event."
        # Get params
        if update_params:
            self._update_vis_params(vrange=vrange,bins=bins,fov=fov,preprocess=preprocess)
        vrange = vrange if vrange is not None else self._vis_params['vrange']
        bins = bins if bins is not None else self._vis_params['bins']
        fov = fov if fov is not None else self._vis_params['fov']
        preprocess = preprocess if preprocess is not None else self._vis_params['preprocess']
        # get frame
        if preprocess is False:
            frame = self.frame
        elif preprocess == 'bksb':
            frame = self.get_frame_bksb_current()
        elif preprocess == 'denoise':
            frame = self.get_frame_denoised_current()
        elif preprocess == 'denoise_nonorm':
            frame = self.get_frame_denoised_current(normalize=False)
        else:
            raise Exception(f"`preprocess` must be in ('bksb', 'denoise', 'denoise_nonorm', False), not {preprocess}")
        # show
        fig,ax = self.show_frame(
            frame,
            vrange=vrange,
            hist=hist,
            bins=bins,
            zoom=zoom,
            fov=fov,
            returnfig=True,
        )
        if returnfig:
            return fig,ax
        else:
            plt.show()

    def show_frame_statistics(self, frame, imageStats, vrange=None, fov=None, profiles=True, update_params=True, returnfig=False):
        """
        """
        # Get params
        if update_params:
            self._update_vis_params(vrange=vrange,fov=fov)
        vrange = vrange if vrange is not None else self._vis_params['vrange']
        fov = fov if fov is not None else self._vis_params['fov']
        # Show
        fig,(ax1,ax2) = self.show_frame(
            frame,
            vrange=vrange,
            fov=fov,
            hist=False,
            returnfig = True
        )
        ax2.plot(self.x,imageStats['yCOMslice'],color='r',label='CoM')
        ax2.plot(self.x,imageStats['yRMSslice']+imageStats['yCOM']-100,color='g',label='S.D.')
        if profiles:
            xProfMax = fov[1]/5/np.max(imageStats['xProfile'])
            yProfMax = fov[0]/5/np.max(imageStats['yProfile'])
            xProfOffset = imageStats['yCOM']-fov[1]/2
            yProfOffset = imageStats['xCOM']-fov[0]/2
            ax2.plot(self.x,xProfMax*imageStats['xProfile']+xProfOffset,color='b',label='x-profile')
            ax2.plot(yProfMax*imageStats['yProfile']+yProfOffset,self.y,color='w',label='y-profile')
        plt.legend()
        if returnfig:
            return fig,(ax1,ax2)
        else:
            plt.show()
            pass

    def show_pulse_statistics(self,idx,n_pulses=None,vrange=None,fov=None,profiles=True,returnfig=False):
        """
        """
        # Get number of pulses
        if n_pulses is not None:
            self._set_n_pulses(n_pulses)
        else:
            n_pulses = self.n_pulses

        # Get frames
        jdx = self._times_good[idx]
        evt = self.get_event(jdx)
        self.set_current_event(evt)
        im = self.get_frame_denoised_current()
        ims,ok = self.split_frame(im,n_pulses)
        assert(ok), "Failed to split image frame"

        # Get stats
        imageStats = []
        for i in range(n_pulses):
            imageStats.append(self._pulse_statistics[i][idx])

        # Show
        figs=[]
        for i in range(n_pulses):
            fig,axs = self.show_frame_statistics(ims[i],imageStats[i],vrange=vrange,fov=fov,profiles=profiles,returnfig=True)
            fig.suptitle(f'Pulse {i+1}')
            figs.append((fig,axs))

        # Return
        if returnfig:
            return figs
        else:
            plt.show()

    def show_denoise_statistics_ROI(self, vrange=None, returnfig=False):
        """
        Parameters
        ----------
        vrange : 2-tuple
            (vmin,vmax)
        returnfig : bool
            toggle returning the plot
        """
        # Get ROI
        _,_,roi = self.get_frame_denoised_current(
                noise_x=noise_x,
                noise_y=noise_y,
                noise_x_end=noise_x_end,
                noise_y_end=noise_y_end,
                returnall=True)
        # Get current frame
        im = self.frame
        assert(im is not None), "Current frame is empty!"
        # Make figure
        if vrange is None:
            vrange = (np.min(im),np.max(im))
        fig,axs = plt.subplots(2,2,figsize=(10,10))
        ((ax11,ax12),(ax21,ax22)) = axs
        ax11.matshow(im,vmin=vrange[0],vmax=vrange[1])
        ax12.matshow(roi)
        ax21.matshow(im*roi,vmin=vrange[0],vmax=vrange[1])
        ax22.matshow(im*np.logical_not(roi),vmin=vrange[0],vmax=vrange[1])
        ax11.invert_yaxis()
        ax12.invert_yaxis()
        ax21.invert_yaxis()
        ax22.invert_yaxis()
        if returnfig:
            return fig,axs
        else:
            plt.show()

    @staticmethod
    def get_plot_lims(x,L):
        """ Get limits for a plot crop window in 1D

        Parameters
        ----------
        x : number
            The window center
        L : number
            The window FOV

        Returns
        -------
        len-2 array of ints : (min, max)
        """
        return np.array([int(np.round(x-L/2)),int(np.round(x+L/2))])





    ##### Helper Utilities #####

    def _progress(self, iterable, desc="", total=None):
        if self._verbose:
            return tqdm(iterable, desc=desc, total=total)
        else:
            return iterable


########## END CLASS ##########
