import numpy as np

import prtools

# function to convert between defocus and z-translation
# pv_z4 = dz/(8*fnum**2)

# function to convert between pv and rms defocus

# function to convert between pv tip/tilt and focal plane position

def radial_avg(a, center=None):

    # https://stackoverflow.com/a/21242776
    
    a = np.asarray(a)

    if center is None:
        r, c = prtools.centroid(a)
    else:
        r, c = center

    rr, cc = np.indices((a.shape))
    rho = np.sqrt((rr-r)**2 + (cc-c)**2).astype(int)

    tbin = np.bincount(rho.ravel(), a.ravel())
    nr = np.bincount(rho.ravel())

    return tbin/nr

def min_sampling(wave, z, du, npix, min_q):
    return (np.min(wave) * z)/(min_q * du * npix)


def pixelscale_nyquist(wave, f_number):
    """Compute the output plane sampling which is Nyquist sampled for
    intensity.

    Parameters
    ----------
    wave : float
        Wavelength in meters

    f_number : float
        Optical system F/#

    Returns
    -------
    float
        Sampling in meters

    """
    return f_number * wave / 2
