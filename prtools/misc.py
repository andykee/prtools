import numpy as np

import prtools


def calcpsf(amp, opd, wavelength, sampling, shape, oversample=2, 
            shift=(0,0), offset=(0,0), weight=1, flatten=True):
    
    sampling = np.broadcast_to(sampling, (2,))
    shape = np.broadcast_to(shape, (2,))
    wavelength = np.atleast_1d(wavelength)
    weight = np.broadcast_to(weight, wavelength.shape)
    shift = np.asarray(shift)

    shape_out = (shape[0]*oversample, shape[1]*oversample)

    out = []
    for wl, wt in zip(wavelength, weight):
        alpha = sampling/(wl*oversample)
        p = amp * np.exp(2*np.pi*1j/wl*opd)
        P = prtools.dft2(p, alpha, shape_out, shift*oversample, offset)
        P = np.abs(P)**2
        out.append(P * wt)
    
    if flatten:
        out = np.sum(out, axis=0)
    else:
        out = np.asarray(out)
    
    return out


def translation_defocus(f_number, dz):
    """Compute the peak-to-valley defocus imparted by a given translation
    along the optical axis
    
    Parameters
    ----------
    f_number : float
        Beam F/#
    dz : float
        Translation along optical axis

    Returns
    -------
    float

    """
    return dz/(8*f_number**2)


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


def fft_shape(dx, du, z, wavelength, oversample):
    """Compute FFT pad shape to satisfy requested sampling condition
    
    Parameters
    ----------
    dx : float or tuple of floats
        Physical sampling of pupil plane. If a single value is supplied,
        the pupil is assumed to be uniformly sampled in both row and column.
    du : float or tuple of floats
        Physical sampling of image plane. If a single value is supplied,
        the image is assumed to be uniformly sampled in both row and column.
    z : float
        Propagation distance
    wavelength : float
        Propagation wavelength
    oversample : float
        Number of times to oversample the output plane
    
    Returns
    -------
    shape : tuple of ints
        Required pad shape
    wavelength : float
        True wavelength represented by padded shape

    """
    # Compute pad shape to satisfy requested sampling. Propagation wavelength
    # is recomputed to account for integer padding of input plane
    alpha = _dft_alpha(dx, du, z, wavelength, oversample)
    fft_shape = np.round(np.reciprocal(alpha)).astype(int)       
    prop_wavelength = np.min((fft_shape/oversample * dx * du)/z)
    return fft_shape, prop_wavelength


def _dft_alpha(dx, du, wavelength, z, oversample):
    dx = np.broadcast_to(dx, (2,))
    du = np.broadcast_to(du, (2,))
    return ((dx[0]*du[0])/(wavelength*z*oversample),
            (dx[1]*du[1])/(wavelength*z*oversample))


def min_sampling(wave, z, du, npix, min_q):
    return (np.min(wave) * z)/(min_q * du * npix)


def pixelscale_nyquist(wave, f_number):
    """Compute the output plane sampling which is Nyquist sampled for
    intensity.

    Parameters
    ----------
    wave : float
        Wavelength
    f_number : float
        Optical system F/#

    Returns
    -------
    float

    """
    return f_number * wave / 2
