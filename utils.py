import matplotlib.pyplot as plt
import numpy as np
import h5py
from typing import List, Dict, Union
from scipy import signal
from datetime import datetime, timedelta, timezone
from epicsArch import *
from scipy.interpolate import interp1d
import scipy.io
from IPython import get_ipython
import re

"""
This file contains utilities for performing radial averaging on image data, 
exploring HDF5 file structures, visualizing detector data, and other useful utilities.

Classes:
    RadialAverager: Performs radial averaging on 2D data.

Functions:
    get_tree(f): Lists the full tree of an HDF5 file.
    is_leaf(dataset): Checks if an HDF5 node is a dataset.
    get_leaves(f, saveto, verbose=False): Extracts datasets from an HDF5 file.
    runNumToString(num): Converts a number to a zero-padded string.
    plot_jungfrau(x, y, f, ax=None, shading='nearest', *args, **kwargs): Plots Jungfrau detector counts.
    recalculateDG2IPM(rawDG2Traces,k_start=876, k_end=926): Recalculates the ipm_dg2 variables from the raw dg2 traces.
"""
class RadialAverager(object):

    def __init__(self, q_values, mask, n_bins=101):
        """
        Parameters
        ----------
        q_values : np.ndarray (float)
            For each pixel, this is the momentum transfer value of that pixel.
        mask : np.ndarray (int)
            A boolean (int) array indicating if each pixel is masked (1) or not (0).
        n_bins : int
            The number of bins to employ.
        """

        self.q_values = q_values
        self.mask = mask
        self.n_bins = n_bins

        # Calculate bin width and range
        self.q_range = self.q_values.max() - self.q_values.min()
        self.bin_width = self.q_range / float(n_bins)

        # Assign each pixel to a bin
        self._bin_assignments = np.floor(
            (self.q_values - self.q_values.min()) / self.bin_width
        ).astype(np.int32)

        # Ensure bin assignments fit within the number of bins
        assert self.n_bins >= self._bin_assignments.max() + 1, 'Incorrect bin assignments'

        # Normalization array for each bin
        self._normalization_array = (
            np.bincount(self._bin_assignments.flatten(), weights=self.mask.flatten()) + 1e-100
        ).astype(float)
        self._normalization_array = self._normalization_array[:self.n_bins]
    

    def __call__(self, image):
        """
        Bin pixel intensities by their momentum transfer.
        
        Parameters
        ----------            
        image : np.ndarray
            The intensity at each pixel, same shape as pixel_pos
        Returns
        -------
        bin_centers : ndarray, float
            The q center of each bin.
        bin_values : ndarray, int
            The average intensity in the bin.
        """
        # Check that the image and q_values have the same shape
        if not (image.shape == self.q_values.shape):
            raise ValueError('image and q_values must have the same shape')
        if not (image.shape == self.mask.shape):
            raise ValueError('image and mask must have the same shape')

        # Calculate the weighted average of the image in each bin
        weights = image.flatten() * self.mask.flatten()
        bin_values = np.bincount(self._bin_assignments.flatten(), weights=weights)
        # 
        bin_values /= self._normalization_array

        # Check that the bin values have the correct shape
        assert bin_values.shape[0] == self.n_bins

        return bin_values
    

    @property
    def bin_centers(self):
        return (np.arange(self.n_bins) + 0.5) * self.bin_width + self.q_values.min()

    @property
    def pixel_counts(self):
        return self._normalization_array


def plot_jungfrau(x, y, f, ax=None, shading='nearest', *args, **kwargs):
    """Plot Jungfrau detector counts.

    Parameters
    ----------
    x, y : list of np.ndarray
        Coordinates for each tile of the detector.
    f : list of np.ndarray
        Data to be plotted for each tile.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, uses current axes.
    shading : str, optional
        Shading style for pcolormesh (default: 'nearest').

    Returns
    -------
    pcm : matplotlib.collections.QuadMesh
        The QuadMesh object created.
    """
    if ax is None:
        ax = plt.gca()
    for i in range(8):
        pcm = ax.pcolormesh(x[i], y[i], f[i], shading=shading, *args, **kwargs)
    return pcm

def combineRuns(runNumbers, folder, keys_to_combine, keys_to_sum, keys_to_check, verbose=False, archImport=True):
    """Combine data from multiple runs into a single dataset.

    Parameters
    ----------
    runNumbers : list of int
        List of run numbers to combine.
    folder : str
        Path to the folder containing the data files.
    verbose : bool, optional
        If True, print detailed information during processing (default: False).

    Returns
    -------
    data_combined : dict
        Dictionary containing combined data from all runs.
    """
    data_array = []
    experiment = folder.split('/')[6]
    for i,runNumber in enumerate(runNumbers):
        data = {}
        filename = f'{folder}{experiment}_Run{runNumToString(runNumber)}.h5'
        print('Loading: ' + filename)
        with h5py.File(filename,'r') as f:
            get_leaves(f,data,verbose=verbose)
            data_array.append(data)
    data_combined = {}
    for key in keys_to_combine:
        # Special routine for loading the gas cell pressure
        epicsLoad = False # Default flag value
        if (key == 'epicsUser/gasCell_pressure') & (archImport):
            try:
                arr = np.squeeze(data_array[0][key])
                for data in data_array[1:]:
                    arr = np.concatenate((arr,np.squeeze(data[key])),axis=0)
                data_combined[key] = arr
            except:
                epicsLoad = True # Set flag if we can't load from the files
        else: # All other keys load normally
            arr = np.squeeze(data_array[0][key])
            for data in data_array[1:]:
                arr = np.concatenate((arr,np.squeeze(data[key])),axis=0)
            data_combined[key] = arr
    run_indicator = np.array([])
    for i,runNumber in enumerate(runNumbers):
        run_indicator = np.concatenate((run_indicator,runNumber*np.ones_like(data_array[i]['lightStatus/xray'])))
    data_combined['run_indicator'] = run_indicator
    for key in keys_to_sum:
        arr = np.zeros_like(data_array[0][key])
        for data in data_array:
            arr += data[key]
        data_combined[key] = arr
    for key in keys_to_check:
        arr = data_array[0][key]
        for i,data in enumerate(data_array):
            if not np.array_equal(data[key],arr):
                print(f'Problem with key {key} in run {runNumbers[i]}')
        data_combined[key] = arr
    # Now to do the special gas cell pressure loading if the flag was set
    if epicsLoad:
        archive = EpicsArchive()
        unixTime = data_combined['unixTime']
        epicsPressure = np.array([]) # Init empty array
        for i,runNumber in enumerate(runNumbers):
            # Pull out start and end times from each run
            runUnixTime = unixTime[run_indicator==runNumber]
            startTime = runUnixTime[0]
            endTime = runUnixTime[-1]
            [times,pressure] = archive.get_points(PV='CXI:MKS670:READINGGET', start=startTime, end=endTime,unit="seconds",raw=True,two_lists=True); # Make Request
            # Interpolate the data
            interp_func = interp1d(times, pressure, kind='previous', fill_value='extrapolate')
            epicsPressure = np.append(epicsPressure,interp_func(runUnixTime)) # Append the data
        # Once all the data is loaded in
        data_combined['epicsUser/gasCell_pressure'] = epicsPressure # Save to the original key.       
    print('Loaded Data')
    return data_combined

def get_tree(f):
    """List the full tree of the HDF5 file.

    Parameters
    ----------
    f : h5py.File
        The HDF5 file object to traverse.

    Returns
    -------
    None
        Prints the structure of the HDF5 file.
    """
    def printname(name):
        print(name, type(f[name]))
    f.visit(printname)
    
def is_leaf(dataset):
    """Check if an HDF5 node is a dataset (leaf node).

    Parameters
    ----------
    dataset : h5py.Dataset or h5py.Group
        The HDF5 node to check.

    Returns
    -------
    bool
        True if the node is a dataset, False otherwise.
    """
    return isinstance(dataset, h5py.Dataset)

def get_leaves(f, saveto, verbose=False):
    """Retrieve all leaf datasets from an HDF5 file and save them to a dictionary.

    Parameters
    ----------
    f : h5py.File
        The HDF5 file object to traverse.
    saveto : dict
        Dictionary to store the retrieved datasets.
    verbose : bool, optional
        If True, print detailed information about each dataset (default: False).

    Returns
    -------
    None
        The datasets are stored in the provided dictionary.
    """
    def return_leaf(name):
        if is_leaf(f[name]):
            if verbose:
                print(name, f[name][()].shape)
            saveto[name] = f[name][()]
    f.visit(return_leaf)

def runNumToString(num):
    """Convert a run number to a zero-padded string of length 4.

    Parameters
    ----------
    num : int
        The run number to convert.

    Returns
    -------
    numstr : str
        The zero-padded string representation of the run number.
    """
    numstr = str(num)
    while len(numstr) < 4:
        numstr = '0' + numstr
    return numstr

def recalculateDG2IPM(rawDG2Traces,k_start=876, k_end=926):
    """Recalculate the ipm_dg2 variables given the raw DG2 traces.

    Parameters
    ----------
    rawDG2Traces : np.ndarray (float)
        Raw DG2 traces.
    k_start : int
        Start index of the maximum search, default is 876
    k_end : int
        End index of the maximum search, default is 926

    Returns
    -------
    sum : np.ndarray (float) 
        Sum of the DG2 readings. Analagous to ipm_dg2/sum.
    xpos : np.ndarray (float)
        X position of the xray beam on the DG2 IPM, in mm
    ypos : np.ndarray (float)
        Y position of the xray beam on the DG2 IPM, in mm
    peaks : np.ndarray (float)
        Peak heights of the traces after applying the fast impulse response filter
    """
    # Fast Impulse Response filter coefficients
    fir = [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 
           0, 0, 0, 0, 0, 0, 
           -0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.125, -0.125]
    filtered_trace = -signal.lfilter(fir, 1, rawDG2Traces, axis=-1) # Filtering the traces
    peaks = np.max(np.abs(filtered_trace[:, :, k_start:k_end]), axis=2) # Calculating all of the peak values
    Cx = -4.79 # X Calibration constant, in % per mm
    Cy = -5.12 # Y Calibration constant, in % per mm
    xpos = (100*(peaks[:,1]-peaks[:,3])/(peaks[:,1]+peaks[:,3]))/Cx
    ypos = (100*(peaks[:,2]-peaks[:,4])/(peaks[:,2]+peaks[:,4]))/Cy
    sums = peaks.sum(axis=1)
    return sums, xpos, ypos, peaks

def hist2dLinFit(xdata,ydata,bins, ax=None,linfit=False,xFrac=None,xVal=None):
    if ax is None:
        ax = plt.gca()
    if linfit:
        if xFrac==None:
            if xVal==None:
                # Assume total data set for fit
                xVal = xdata.max()
            # Do the cropping based on the value
            xCrop = xdata[xdata<=xVal]
            yCrop = ydata[xdata<=xVal]
        else:
            # Do the cropping based on the fraction of the max
            xCrop = xdata[xdata/xdata.max()<=xFrac]
            yCrop = ydata[xdata/xdata.max()<=xFrac]
        poly_coeffs = np.polyfit(xCrop, yCrop, 1)
        print(poly_coeffs)
        # Generate fit curve
        x_fit = np.linspace(np.min(xdata), np.max(xdata), 100)
        y_fit = np.polyval(poly_coeffs, x_fit)
        y_lin = np.polyval(poly_coeffs, xdata)
        residuals = ydata-y_lin
    
        rcoeff = scipy.stats.pearsonr(xCrop, yCrop)
        plt.text(0.1, 0.8, f'Pearson r coeff: {rcoeff.statistic:.9f}',
                 fontsize=12, fontweight='bold', transform=plt.gca().transAxes, ha='left', color='White')
        # Overlay the fit curve
        plt.plot(x_fit, y_fit, color='red', linewidth=2, label='Linear Fit', linestyle='--',zorder=2)
        plt.legend()
    
    plt.hist2d(xdata,ydata,bins, zorder=1);
    return residuals


def enable_underscore_cleanup():
    """Registers a post-cell hook to delete user-defined _ variables after each cell."""
    ipython = get_ipython()
    user_ns = ipython.user_ns  # This gives you access to the Jupyter notebook namespace

    def clean_user_underscore_vars(*args, **kwargs):
        def is_user_defined_underscore(var):
            return (
                var.startswith('_')
                and not re.match(r'^_i\d*$|^_\d*$|^_ih$|^_oh$|^_ii*$|^_iii$|^_dh$|^_$', var)
                and not var.startswith('__')
            )

        for var in list(user_ns):
            if is_user_defined_underscore(var):
                del user_ns[var]

    ipython.events.register('post_run_cell', clean_user_underscore_vars)
