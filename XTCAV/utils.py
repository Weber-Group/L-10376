import numpy as np
from scipy.ndimage import binary_closing
import psana
from xtcav.ShotToShotCharacterization import ShotToShotCharacterization

def load_xtcav_run_first_pulse(runnum):
    """
    Load the first recorded XTCAV pulse from a data run.

    Parameters
    ----------
    runnum : integer
        The run number

    Return
    ------
    (ar,bool) : the image array, and a success/failure flag
    """
    # data source
    dataSource=psana.DataSource("exp=cxil1037623:run=%d:idx"%runnum)

    # XTCAV data shot-by-shot retrieval object
    XTCAVRetrieval=ShotToShotCharacterization();
    XTCAVRetrieval.SetEnv(dataSource.env())

    # get the run and shot timing data
    run = next(dataSource.runs())
    times = run.times()

    # find the first event with XTCAV data
    t_idx = 0
    success = False
    while not success and t_idx<1000:
        t = times[t_idx]
        evt = run.event(t)
        success = XTCAVRetrieval.SetCurrentEvent(evt)
        #print(f"t={t_idx}: success = {success}")
        t_idx += 1

    # get the XTCAV camera raw image data
    image,flag=XTCAVRetrieval.RawXTCAVImage()

    # return
    return image,flag


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
    yy,xx = np.meshgrid(np.arange(s[1]),np.arange(s[0]))
    m = np.sum(ar)
    x = np.sum(xx*ar)/m
    y = np.sum(yy*ar)/m
    return np.array([x,y])

def get_xtcav_mask(ar, thresh=None, thresh_mode='median', std=1, close=1):
    """
    Get a mask for an xtcav camera acquisition.

    Pixel values below a threshold are masked, then a binary closing operation is performed.
    The threshold can be set manually with the `thresh` parameter, or will be automatically
    set to AVE + N*STD, where AVE is the median or mean (set by `thresh_mode`), STD is the
    standard deviation, and N a number (set by `std`).

    Parameters
    ----------
    ar : 2d array
        The image
    thresh : number or None
        The threshold value.  If None, set automatically
    thresh_mode : 'median' or 'mean'
        Sets how AVE is determined
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
            thresh = np.median(ar)
        else:
            thresh = np.mean(ar)
        # add std
        thresh += np.std(ar) * std
    # get mask
    mask = ar<thresh
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
    
def get_com_clean(ar, thresh=None, thresh_mode='median', std=1, close=2, returnmask=False):
    """
    Return the 2D center of mass, ignoring certain masked pixel values.
    See `get_xtcav_mask` docstring for masking details.

    Parameters
    ----------
    ar : 2d array
        The object
    thresh : number or None
        See `get_xtcav_mask` for details
    thresh_mode : 'median' or 'mean'
        See `get_xtcav_mask` for details
    std : number
        See `get_xtcav_mask` for details
    close : int
        See `get_xtcav_mask` for details
    returnmask : bool
        Toggles returning the mask with the answer. If true, return value
        becomes a 2-tuple ([x,y],mask)

    Returns
    -------
    [x,y] or ([x,y],mask)
    """
    # get mask
    mask = get_xtcav_mask(ar,thresh,thresh_mode,std,close)
    # compute & return
    _ar = np.copy(ar)
    _ar[mask] = 0
    ans = get_com(_ar)
    if returnmask:
        return ans, mask
    else:
        return ans

def get_plot_lims(x,L):
    return np.array([int(np.round(x-L/2)),int(np.round(x+L/2))])
