import numpy as np

import prtools


def encircled_energy(a, energy=0.8, center=None):
    """Compute the encircled energy diameter for a given energy fraction.

    Parameters
    ----------
    a : array_like
        Input array
    energy : float or array_like, optional
        Encircled energy fraction (0 < energy < 1). Default is 0.8. Can also
        be a vector. 
    center : (2,) array_like or None
        Coordinates of center of circle given as (row, col). If None 
        (default), the center coordinates are computed as the centroid of a.
    Returns
    -------
    diam : float or ndarray
        Diameter in pixels of circle enclosing specified energy fraction.
    ee : float or ndarray
        Encircled energy at computed diameter.

    """

    a = np.asarray(a)

    if a.shape[0] == a.shape[1]:
        npix = a.shape[0]
    else:
        npix = np.max(a.shape)
        a = prtools.pad(a, (npix,npix))
    
    if center is None:
        yc, xc = prtools.centroid(a)
    else:
        yc, xc = center

    x = np.arange(npix) - xc
    y = np.arange(npix) - yc

    xx, yy = np.meshgrid(x, y)

    r = np.abs(xx + 1j*yy)

    # initialize some variables    
    ee_out = []
    d_out = []
    etot = np.sum(a)

    energy = np.atleast_1d(energy)

    #rad = 0
    #ee = 0
    #ee0 = -1

    for e in energy:
        rad = 0
        ee = 0
        ee0 = -np.inf
        
        # this seemingly bizarre bit of math establishes the evolution 
        # of the step size during the search
        #     * dfactor is the factor dpix is reduced by when the step 
        #       size is updated
        #     * dpix is the initial starting guess for the inner radius
        dfactor = 4
        dpix = dfactor**np.floor(np.log(npix)/np.log(dfactor) - 1)

        while ee != ee0:
            while ee < e and ee != ee0:
                rad = rad + dpix
                emasked = (r < rad) * a
                ee0 = ee
                ee = np.sum(emasked)/etot
                #print(f'Inner rad = {rad}, ee = {ee}')

            rad = rad - dpix
            dpix = dpix/dfactor
            rad = rad + dpix
            emasked = (r < rad) * a
            ee0 = ee
            ee = np.sum(emasked)/etot
            #print(f'Outer rad = {rad}, ee = {ee}')
        
        ee_out.append(ee)
        d_out.append(2*rad)

    if energy.size == 1:
        ee_out = ee_out[0]
        d_out = d_out[0]

    return d_out, ee_out


def pv(a, axis=None):
    """Compute peak-to-valley or max(a) - min(a)
    
    Parameters
    ----------
    a : array_like
        Input array
    axis: None or int, optional
        Axis or axes along which the peak-to-valley is computed. The 
        default is to compute the peak-to-valley of the flattened 
        array.

    Returns
    -------
    pv : ndarray
        A new array containing the P-V along the requested axes.
    """
    a = np.asarray(a)
    return np.amax(a, axis=axis) - np.min(a, axis=axis)


def rms(a, axis=None):
    """Compute the root-mean-square of the nonzero entries

    Parameters
    ----------
    a : array_like
        Input array
    axis: None or int, optional
        Axis or axes along which the standard deviation is computed. The 
        default is to compute the standard deviation of the flattened 
        array.

    Returns
    -------
    rms : ndarray
        A new array containing the RMS along the requested axes.

    """
    a = np.asarray(a)
    return np.std(a[np.nonzero(a)], axis=axis)


# A good way to measure strehl is through MTF. Strehl is the ratio of the 
# integral of whatever your MTF is to the integral of the diffraction limited 
# MTF.
#
# Because MTF is normalized, the "bulk flux" so to speak (DC component) is 
# removed as something that can produce error in your measurement.
#
# The low frequencies carry a "large amount" of the energy in the MTF, and to 
# accurately measure them you need a pretty large "field of view" of the 
# PSF -- many airy radii (many meaning several tens of them, say ~50 of them).
#
# N.b., no free lunch; it removes any notion of "DC accuracy" but requires 
# good knowledge of the F/# (really, aperture shape) and wavelength so you can 
# compute the diffraction limited MTF.
#def strehl(a):
#    pass
