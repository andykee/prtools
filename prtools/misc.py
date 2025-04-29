import prtools
from prtools._backend import numpy as np


def calcpsf(amp, opd, wavelength, sampling, shape, oversample=2,
            shift=(0, 0), offset=(0, 0), weight=1, flatten=True):
    r"""Calculate a point spread function using far-field diffraction.

    Parameters
    ----------
    amp : array_like
        Pupil amplitude
    opd : array_like
        Pupil OPD
    wavelength : float or list_like
        Propagation wavelength(s)
    sampling : float or tuple of floats
        Propagation sampling term defined as

        .. math::
            \mbox{sampling} = \\frac{dx \ du}{z}

        where *dx* is the pupil plane spatial sampling, *du* is the image
        plane spatial sampling, and *z* is the propagation distance. If a
        single value is supplied, the sampling is assumed to be uniform
        in both row and column.
    shape : int or tuple of ints
        Native output shape. If a single value is supplied, the output will
        have shape = (shape, shape). Note that the actual output shape is
        floor(shape * oversample)
    oversample : float
        Number of times to oversample the output plane
    shift : tuple of floats, optional
        Shift from (0,0) in (r,c) of image plane. Default is (0,0).
    offset : tuple of floats, optional
        Offset from (0,0) in (r,c) of pupil plane. Default is (0,0).
    weight : float or list_like
        Weighting for each wavelength in ``wavelength``.
    flatten : bool, optional
        If True (default),the output is flattened along the spectral
        dimension. Otherwise, a cube is returned with shape
        (len(wavelength), shape[0]*oversample, shape[1]*oversample)

    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> amp = prtools.circle(shape=(256, 256), radius=100)
        >>> coeffs = np.random.uniform(low=-1, high=1, size=8)*1e-7
        >>> opd = prtools.zernike_compose(amp, coeffs)
        >>> wave = 500e-9
        >>> dx = 1/200  # pupil diameter = 1m
        >>> du = 5e-6
        >>> focal_length = 10
        >>> sampling = (dx * du)/focal_length
        >>> shape = (64, 64)
        >>> oversample = 5
        >>> psf = prtools.calcpsf(amp, opd, wave, sampling, shape, oversample)
        >>> plt.imshow(psf, cmap='inferno', norm='log', vmin=10e-5)

    """
    sampling = np.broadcast_to(sampling, (2,))
    shape = np.broadcast_to(shape, (2,))
    wavelength = np.atleast_1d(wavelength)
    weight = np.broadcast_to(weight, wavelength.shape)
    shift = np.asarray(shift)

    shape_out = tuple(np.floor((shape[0]*oversample, shape[1]*oversample)).astype(int))

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

def fft_shape(dx, du, z, wavelength, oversample):
    """Compute FFT pad shape to satisfy requested sampling condition

    Parameters
    ----------
    dx : float or tuple of floats
        Spatial sampling of pupil plane. If a single value is supplied,
        the pupil is assumed to be uniformly sampled in both row and column.
    du : float or tuple of floats
        Spatial sampling of image plane. If a single value is supplied,
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
    """Compute the minimum pupil plane sampling to satisfy given
    constraints.

    """
    return (np.min(wave) * z)/(min_q * du * npix)


def pixelscale_nyquist(wave, f_number):
    """Compute the output plane sampling that is Nyquist sampled for
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


def find_wrapped(opd, grad_thresh):
    """Attempt to find phase wrapping using gradient edge detection while
    automatically ignoring large gradients at mask edges.

    Parameters
    ----------
    opd : array_like
        OPD to detect wrapping in
    grad_thresh : float
        Threshold on magnitude of OPD gradient for determining wrapping

    Returns
    -------
    wrapped : ndarray
        Map of wrapped pixels
    grad : ndarray
        Gradient amplitude

    """
    opd = np.asarray(opd)

    # compute gradient magnitude
    gr, gc = np.gradient(opd)
    G = np.sqrt(gr**2 + gc**2)

    # compute mask to ignore gradient boundary
    mask = np.zeros_like(opd)
    mask[np.where(opd != 0)] = 1
    grm, gcm = np.gradient(mask)
    Gm = np.zeros_like(G)
    Gm[np.where(np.sqrt(grm**2 + gcm**2) == 0)] = 1

    grad = G * Gm
    wrapped = np.zeros_like(grad)
    wrapped[np.where(grad > grad_thresh)] = 1

    return wrapped, grad
