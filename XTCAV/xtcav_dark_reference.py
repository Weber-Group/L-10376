# Dark Reference Class

import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from past.utils import old_div
from xtcav_processor import XTCAVProcessor

class XTCAVDarkReference(XTCAVProcessor):

    def __init__(self, data_source, max_shots=401, compute=True, env=None, iteration='frames', verbose=True, _test_xy=False, _preindex=False):
        """
        Parameters
        ----------
        data_source : a psana DataSource instance
            Must point to a dark run
        max_shots : integer
            Maximum number of camera frames to average
        compute : bool
            If true, compute the dark reference upon instantiation
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
        self._max_shots = max_shots
        self._dark_reference = None
        if compute:
            self.compute()
        pass

    @property
    def max_shots(self):
        return self._max_shots
    
    def compute(self):
        """ Calculate the dark reference
        """
        # setup
        ans = np.zeros(self.frameshape, dtype=np.float64)
        n = 0

        # loop
        progbar = tqdm(desc="Calculating dark background reference", total=self.max_shots)
        for i,j,evt in self.shot_iterator:
            # finish?
            if n >= self.max_shots:
                progbar.n = n
                progbar.refresh()
                progbar.close()
                break             
            # set event and get camera image
            self.set_current_event(evt)
            im = self.frame
            # skip if empty image
            if im is None: 
                continue
            # add
            ans += im
            n += 1
            # update progress
            if n%10==0:
                progbar.n = n
                progbar.refresh()


        # finish
        self._dark_reference=old_div(ans,n)
        
        # print       
        if self._verbose:
            #At the end of the program the total accumulator is saved 
            print('\nMaximum number of images processed\n') 
        
    @property
    def dark_reference(self):
        return self._dark_reference

    def show(self, vrange=None, hist=True, bins=None, zoom=True, fov=(120,240), returnfig=False):
        ans = self.show_frame(
            frame = self.dark_reference,
            vrange = vrange,
            hist = hist,
            bins = bins,
            zoom = False,
            fov = fov,
            returnfig = returnfig
        )
        if returnfig:
            return ans
        else:
            plt.show()

    # TODO - save method, paths
    # TODO - calib parameters

    # def save(self):
    #     cp = CalibrationPaths(dataSource.env(), self.parameters.calibration_path)
    #     file = cp.newCalFileName(Cn.DB_FILE_NAME, self.parameters.validity_range[0], self.parameters.validity_range[1])
    #     self.save(file)




