import numpy as np

import prtools


def ee(a, energy=0.8, center=None):
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
        a = prtools.pad(a, (npix, npix))

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
                # print(f'Inner rad = {rad}, ee = {ee}')

            rad = rad - dpix
            dpix = dpix/dfactor
            rad = rad + dpix
            emasked = (r < rad) * a
            ee0 = ee
            ee = np.sum(emasked)/etot
            # print(f'Outer rad = {rad}, ee = {ee}')

        ee_out.append(ee)
        d_out.append(2*rad)

    if energy.size == 1:
        ee_out = ee_out[0]
        d_out = d_out[0]

    return d_out, ee_out


def pv(a, axis=None):
    """Compute range of values (max - min)

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
    ndarray

    See also
    --------
    :func:`nanpv`

    """
    a = np.asarray(a)
    return np.max(a, axis=axis) - np.min(a, axis=axis)


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
    ndarray

    See also
    --------
    :func:`nanrms`

    """
    a = np.asarray(a)
    return np.std(a[np.nonzero(a)], axis=axis)


def nanpv(a, axis=None):
    """Compute range of values (max - min), ignoring any NaNs.

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
    ndarray

    See also
    --------
    :func:`pv`

    """
    a = np.asarray(a)
    return np.nanmax(a, axis=axis) - np.nanmin(a, axis=axis)


def nanrms(a, axis=None):
    """Compute the root-mean-square of the nonzero entries, ignoring any NaNs.

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
    ndarray

    See also
    --------
    :func:`rms`

    """
    a = np.array(a, copy=True)
    a[np.isnan(a)] = 0
    return rms(a, axis)


def radial_avg(a, center=None):
    """Compute the average radial profile of the input

    Parameters
    ----------
    a : array_like
        Input array
    center : (2,) array_like or None
        Coordinates of center of ``a`` given as (row, col). If None (default),
        the center coordinates are computed as the centroid of ``a``.

    Returns
    -------
    ndarray

    References
    ----------
    [1] https://stackoverflow.com/a/21242776

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> amp = prtools.circle(shape=(256,256), radius=100)
        >>> psf = prtools.calcpsf(amp, 0, 500e-9, 1e-9, (64,64), 3)
        >>> psf /= np.max(psf)
        >>> ra = prtools.radial_avg(psf**0.1)
        >>> fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(5,2))
        >>> ax1.set_title('a')
        >>> ax1.imshow(psf**0.1)
        >>> ax2.set_title('radial_avg(a)')
        >>> ax2.plot(np.arange(ra.size), ra)
        >>> ax2.grid('on')

    """
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
