from math import factorial
import warnings

import numpy as np

import prtools
from prtools.shape import _shift_ij


def _warn_legacy(legacy):
    # Called once at the top of each public function. stacklevel=3 skips this
    # helper and the public function so the warning points at the caller.
    if legacy:
        warnings.warn(
            'legacy=True reproduces the Zernike orientation convention of '
            'prtools 2.0.2 and earlier. Support for it may be removed in a '
            'future release.',
            FutureWarning,
            stacklevel=3
        )


def zernike(mask, index, normalize=True, order='noll', rho=None, theta=None,
            legacy=False):
    """Compute the circular Zernike polynomial for a given mask.

    Parameters
    ----------
    mask : array_like
        Binary mask defining the extent to compute the Zernike polynomial
        over.
    index : int
        Zernike mode index
    normalize : bool, optional
        If True (default), the output is normalized according to [1]. If False,
        the output value ranges [-1, 1] over the mask.
    order : {'noll', 'fringe', 'ansi'}, optional
        Zernike ordering scheme. Default is 'noll'.
    rho : array_like, optional
        Radial coordinates of the mask array. :attr:`rho` should be 0 at the
        origin and 1 at the edge of the circle.
    theta : array_like, optional
        Angular coordinates of the mask array in radians.
    legacy : bool, optional
        If True, reproduce the Zernike convention of prtools 2.0.2 and
        earlier. Default is False. See the warning below for details.

    Returns
    -------
    ndarray

    Warnings
    --------
    prtools 2.0.2 and earlier computed Zernike polynomials in a coordinate
    system mirrored top-to-bottom (y -> -y) relative to the current one. This
    results in modes with negative azimuthal order (sine terms: y tilt, 45
    degree astigmatism, y coma, ...) having a flipped sign. Modes with zero
    or positive azimuthal order are unaffected. The
    :func:`zernike_coeff_legacy` function converts between current and legacy
    conventions.

    Notes
    -----
    Zernike polynomials are defined to be orthogonal on the unit circle. If
    the supplied mask is non-circular, the Zernike polynomial is computed on an
    outscribing circle and then cropped by the mask. Note that this operation
    breaks the orthogonality of the Zernike polynomial. When working with a
    non-circular mask, care must be taken to understand any side-effects of
    using Zernike polynomials constructed in this manner. Any undesirable
    side-effects can be mitigated by using Zernike polynomials that have
    undergone Gram-Schmidt orthogonalization over the supplied mask.

    References
    ----------
    [1] Noll, RJ. Zernike polynomials and atmospheric turbulence. J Opt Soc Am 66, 207-211  (1976).

    """
    _warn_legacy(legacy)
    return _zernike(mask, index, normalize, order, rho, theta, legacy)


def _zernike(mask, index, normalize, order, rho, theta, legacy):
    mask = np.asarray(mask, dtype=bool)

    if order == 'noll':
        n, m = noll_to_nm(index)
    elif order == 'fringe':
        n, m = fringe_to_nm(index)
    elif order =='ansi':
        n, m = ansi_to_nm(index)
    else:
        raise ValueError('Unknown order', order)
    
    if rho is None:
        if legacy:
            rho, theta = _zernike_coordinates_legacy(mask)
        else:
            rho, theta = zernike_coordinates(mask)
    else:
        if theta is None:
            raise ValueError("Both rho and theta must be specified")

    if m == 0:
        if n == 0:
            Z = mask.astype(float)
        else:
            if normalize:
                Z = np.sqrt(n+1) * R(m, n, rho) * mask
            else:
                Z = R(m, n, rho) * mask

    elif m > 0:
        if normalize:
            Z = np.sqrt(2) * np.sqrt(n+1) * R(m, n, rho) * np.cos(m*theta) * mask
        else:
            Z = R(m, n, rho) * np.cos(m*theta) * mask

    else:
        if legacy:
            sin_m_theta = np.sin(m*theta)
        else:
            sin_m_theta = np.sin(np.abs(m)*theta)

        if normalize:
            Z = np.sqrt(2) * np.sqrt(n+1) * R(m, n, rho) * sin_m_theta * mask
        else:
            Z = R(m, n, rho) * sin_m_theta * mask

    #out[mask_slice] = Z
    out = Z
    return out

def R(m, n, rho):

    m = int(np.abs(m))
    n = int(np.abs(n))

    if (n - m) & 1:  # odd
        return 0
    else:
        R = np.zeros(rho.shape)
        for k in range(int(n-m)//2 + 1):
            Rk = ((-1) ** k * factorial(n-k) /
                  (factorial(k) * factorial((n+m)//2-k) * factorial((n-m)//2-k)))
            R += Rk * rho ** (n-2*k)
        return R
    

def zernike_compose(mask, coeffs, normalize=True, order='noll', rho=None,
                    theta=None, legacy=False):
    """Create an OPD based on the supplied Zernike coefficients.

    Parameters
    ----------
    mask : array_like
        Binary mask defining the extent to compute the Zernike polynomial over.
    coeffs : array_like
        List of coefficients corresponding to Zernike indices used to create
        the OPD.
    normalize : bool, optional
        If True (default), the output is normalized according to [1]. If False,
        the output value ranges [-1, 1] over the mask.
    order : {'noll', 'fringe', 'ansi'}, optional
        Zernike ordering scheme. Default is 'noll'.
    rho : array_like, optional
        Radial coordinates of the mask array. :attr:`rho` should be 0 at the
        origin and 1 at the edge of the circle.
    theta : array_like, optional
        Angular coordinates of the mask array in radians.
    legacy : bool, optional
        If True, reproduce the Zernike convention of prtools 2.0.2 and
        earlier. Default is False. See the warning below for details.

    Returns
    -------
    ndarray

    Warnings
    --------
    prtools 2.0.2 and earlier computed Zernike polynomials in a coordinate
    system mirrored top-to-bottom (y -> -y) relative to the current one. This
    results in modes with negative azimuthal order (sine terms: y tilt, 45
    degree astigmatism, y coma, ...) having a flipped sign. Modes with zero
    or positive azimuthal order are unaffected. The
    :func:`zernike_coeff_legacy` function converts between current and legacy
    conventions.

    Examples
    --------
    Compute a random OPD using the first ten Zernikes:

    .. plot::
        :scale: 50
        :include-source:
        :context: reset

        >>> mask = prtools.circle((256,256), 120, antialias=False)
        >>> coeffs = np.random.rand(10)*1e-8
        >>> opd = prtools.zernike_compose(mask, coeffs)
        >>> plt.imshow(opd)

    Using the same mask, compute an OPD representing 200 nm focus error (Z4) and -100 nm
    astigmatism error (Z6):

    .. plot::
        :context: close-figs
        :include-source:
        :scale: 50

        >>> opd = prtools.zernike_compose(mask, [0, 0, 0, 200e-9, 0, -100e-9])
        >>> plt.imshow(opd)

    References
    ----------
    [1] Noll, RJ. Zernike polynomials and atmospheric turbulence. J Opt Soc Am 66, 207-211  (1976).

    """
    _warn_legacy(legacy)
    return _zernike_compose(mask, coeffs, normalize, order, rho, theta, legacy)


def _zernike_compose(mask, coeffs, normalize, order, rho, theta, legacy):
    mask = np.asarray(mask)
    coeffs = np.asarray(coeffs)
    opd = np.zeros(mask.shape)

    for index, coeff in enumerate(coeffs):
        opd += coeff * _zernike(mask, index+1, normalize, order, rho, theta,
                                legacy)

    return opd

def zernike_basis(mask, modes, vectorize=False, normalize=True, order='noll',
                  rho=None, theta=None, legacy=False):
    """Compute a Zernike basis set for a given mask.

    Parameters
    ----------
    mask : array_like
        Binary mask defining the extent to compute the Zernike polynomial over.
    modes : array_like
        List of modes to return.
    vectorize : bool, optional
        If True, the output is returned as a
        ``(length(modes), modes.shape[0]*modes.shape[1])`` array If False
        (default), the output is returned as a
        ``(length(terms), mask.shape[0], mask.shape[1])`` cube.
    normalize : bool, optional
        If True (default), the output is normalized according to [1]. If False,
        the output value ranges [-1, 1] over the mask.
    order : {'noll', 'fringe', 'ansi'}, optional
        Zernike ordering scheme. Default is 'noll'.
    rho : array_like, optional
        Radial coordinates of the mask array. :attr:`rho` should be 0 at the
        origin and 1 at the edge of the circle.
    theta : array_like, optional
        Angular coordinates of the mask array in radians.
    legacy : bool, optional
        If True, reproduce the Zernike convention of prtools 2.0.2 and
        earlier. Default is False. See the warning below for details.

    Returns
    -------
    ndarray

    Warnings
    --------
    prtools 2.0.2 and earlier computed Zernike polynomials in a coordinate
    system mirrored top-to-bottom (y -> -y) relative to the current one. This
    results in modes with negative azimuthal order (sine terms: y tilt, 45
    degree astigmatism, y coma, ...) having a flipped sign. Modes with zero
    or positive azimuthal order are unaffected. The
    :func:`zernike_coeff_legacy` function converts between current and legacy
    conventions.

    References
    ----------
    [1] Noll, RJ. Zernike polynomials and atmospheric turbulence. J Opt Soc Am 66, 207-211  (1976).

    """
    _warn_legacy(legacy)
    return _zernike_basis(mask, modes, vectorize, normalize, order, rho, theta,
                          legacy)


def _zernike_basis(mask, modes, vectorize, normalize, order, rho, theta,
                   legacy):
    modes = np.asarray(modes)
    if modes.shape == ():
        modes = modes[..., np.newaxis]

    mask = np.asarray(mask)

    basis = []
    for index, mode in enumerate(modes):
        basis.append(_zernike(mask, mode, normalize, order, rho, theta,
                              legacy))
    basis = np.asarray(basis)

    if vectorize:
        # reshape basis from cube to matrix
        return basis.reshape(basis.shape[0], -1)
    else:
        return basis


def zernike_fit(opd, mask, modes, normalize=True, order='noll', rho=None,
                theta=None, legacy=False):
    """Fit a Zernike basis set to an OPD.

    Parameters
    ----------
    opd : array_like
        OPD to fit. NaN values in ``opd`` are excluded from the fit.
    mask : array_like
        Binary mask defining the extent to compute the Zernike basis over.
    modes : array_like
        List of modes to fit.
    normalize : bool, optional
        If True (default), the output is normalized according to [1]. If False,
        the output value ranges [-1, 1] over the mask.
    order : {'noll', 'fringe', 'ansi'}, optional
        Zernike ordering scheme. Default is 'noll'.
    rho : array_like, optional
        Radial coordinates of the mask array. :attr:`rho` should be 0 at the
        origin and 1 at the edge of the circle.
    theta : array_like, optional
        Angular coordinates of the mask array in radians.
    legacy : bool, optional
        If True, reproduce the Zernike convention of prtools 2.0.2 and
        earlier. Default is False. See the warning below for details.

    Returns
    -------
    ndarray

    Warnings
    --------
    prtools 2.0.2 and earlier computed Zernike polynomials in a coordinate
    system mirrored top-to-bottom (y -> -y) relative to the current one. This
    results in modes with negative azimuthal order (sine terms: y tilt, 45
    degree astigmatism, y coma, ...) having a flipped sign. Modes with zero
    or positive azimuthal order are unaffected. The
    :func:`zernike_coeff_legacy` function converts between current and legacy
    conventions.

    Example
    -------
    .. code:: pycon

        >>> mask = prtools.circle((256,256),128)
        >>> coeffs = np.random.rand(4)*100e-9
        >>> opd = prtools.zernike_compose(mask, coeffs)
        >>> fit_coeffs = prtools.zernike_fit(opd, mask, np.arange(2,4))
        >>> print('Tip/tilt coefficients:', coeffs[1:3])
        >>> print('Fit tip/tilt coefficients:', fit_coeffs)

        Tip/tilt coefficients: [9.69097470e-08 9.94332699e-08]
        Fit tip/tilt coefficients: [9.69545890e-08 9.94781119e-08]

    See Also
    --------
    :func:`zernike_remove` Fit and remove a Zernike basis set from an OPD.

    References
    ----------
    [1] Noll, RJ. Zernike polynomials and atmospheric turbulence. J Opt Soc Am 66, 207-211  (1976).

    """
    _warn_legacy(legacy)
    return _zernike_fit(opd, mask, modes, normalize, order, rho, theta, legacy)


def _zernike_fit(opd, mask, modes, normalize, order, rho, theta, legacy):
    opd = np.asarray(opd).ravel()
    mask = np.asarray(mask)
    basis = _zernike_basis(mask, modes, True, normalize, order, rho, theta,
                           legacy)
    
    # valid data includes all non-NaN values inside the mask
    valid = np.isfinite(opd) & mask.ravel().astype(bool)
    
    coeff, *_ = np.linalg.lstsq(basis[:, valid].T, opd[valid], rcond=None)

    return coeff


def zernike_remove(opd, mask, modes, order='noll', rho=None, theta=None,
                   legacy=False):
    """Fit and remove a Zernike basis set from an OPD.

    Parameters
    ----------
    opd : array_like
        OPD to fit.
    mask : array_like
        Binary mask defining the extent to compute the Zernike basis over.
    modes : array_like
        List of modes to remove.
    order : {'noll', 'fringe', 'ansi'}, optional
        Zernike ordering scheme. Default is 'noll'.
    rho : array_like, optional
        Radial coordinates of the mask array. :attr:`rho` should be 0 at the
        origin and 1 at the edge of the circle.
    theta : array_like, optional
        Angular coordinates of the mask array in radians.
    legacy : bool, optional
        If True, reproduce the Zernike convention of prtools 2.0.2 and
        earlier. Default is False. See the warning below for details.

    Returns
    -------
    ndarray

    Warnings
    --------
    prtools 2.0.2 and earlier computed Zernike polynomials in a coordinate
    system mirrored top-to-bottom (y -> -y) relative to the current one. This
    results in modes with negative azimuthal order (sine terms: y tilt, 45
    degree astigmatism, y coma, ...) having a flipped sign. Modes with zero
    or positive azimuthal order are unaffected. The
    :func:`zernike_coeff_legacy` function converts between current and legacy
    conventions.

    See Also
    --------
    :func:`zernike_fit`

    References
    ----------
    [1] Noll, RJ. Zernike polynomials and atmospheric turbulence. J Opt Soc Am 66, 207-211  (1976).

    """
    _warn_legacy(legacy)
    opd = np.asarray(opd)
    mask = np.asarray(mask)

    coeffs = _zernike_fit(opd, mask, modes, True, order, rho, theta, legacy)
    fit_opd = _zernike_compose(mask, coeffs, True, order, rho, theta, legacy)

    residual = opd - fit_opd

    return residual


def noll_to_nm(j):
    """Convert Noll index to (n, m) index.

    Parameters
    ----------
    j : int
        Noll Zernike index

    Returns
    -------
    n : int
        Radial degree.
    m : int
        Azimuthal degree.
    """
    if j < 1:
        raise ValueError('Noll index must be greater than zero')

    # find the row j is in
    n = int(np.ceil((-1 + np.sqrt(1 + 8 * j)) / 2) - 1)
    if n == 0:
        m = 0
    else:
        # find where in the row j is
        k = (n + 1) * (n + 2) / 2
        r = int(j - k - 1)

        if j & 1:  # odd
            sign = -1
        else:
            sign = 1

        if n & 1:  # odd
            row_m = [1, 1]
        else:
            row_m = [0]

        for _ in range(n // 2):
            row_m.append(row_m[-1] + 2)
            row_m.append(row_m[-1])

        m = row_m[r] * sign

    return int(n), int(m)


def ansi_to_nm(j):
    """Convert ANSI index to (n, m) index.

    Parameters
    ----------
    j : int
        ANSI index

    Returns
    -------
    n : int
        Radial degree
    m : int
        Azimuthal degree
    """
    if j < 0:
        raise ValueError('ANSI index must be zero or larger')
    
    # https://en.wikipedia.org/wiki/Zernike_polynomials
    n = np.floor((np.sqrt(8 * j + 1) - 1) / 2)
    m = 2*j - n*(n + 2)
    return int(n), int(m)


def fringe_to_nm(j):
    """Convert Fringe index to (n, m) index.

    Parameters
    ----------
    j : int
        Fringe index

    Returns
    -------
    n : int
        Radial degree
    m : int
        Azimuthal degree
    """
    if j < 1:
        raise ValueError('Fringe index must be greater than zero')
    
    # https://github.com/arielmission-space/PAOS/blob/main/paos/classes/zernike.py
    m_n = 2 * (np.ceil(np.sqrt(j)) - 1)
    g_s = (m_n / 2) ** 2 + 1
    n = m_n / 2 + np.floor((j - g_s) / 2)
    m = (m_n - n) * (1 - np.mod(j - g_s, 2) * 2)
    return int(n), int(m)


def zernike_coordinates(mask, shift=None, angle=0, indexing='ij'):
    """Compute the Zernike coordinate system for a given mask.
    
    Parameters
    ----------
    mask : array_like
        Binary mask defining the extent to compute the Zernike polynomial
        over. Nonzero values are treated as inside the mask.
    shift : (2,) array_like, optional
        How far to shift the origin from the center of the array. (rows,
        cols) for 'ij', (x, y) for 'xy'. If None (default), the
        origin is placed at the mask centroid.
    angle: float, optional
        Angle to rotate coordinate system by in degrees, from the +x axis
        toward the +y axis. Default is 0.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of ``shift``.

    Returns
    -------
    rho, theta : ndarrays
        Normalized radius and angle. theta is measured from +x (increasing
        column) toward +y (increasing row), which is counterclockwise when
        displayed with ``origin='lower'``.

    """
    mask = np.asarray(mask, dtype=bool)

    if shift is None:
        # (row, col) offset of the mask centroid from the mesh origin
        shift = prtools.centroid(mask, mode='center')
    else:
        shift = _shift_ij(shift, indexing)

    # an xy rotation from +x toward +y is a negative ij rotation
    rr, cc = prtools.mesh(shape=mask.shape, shift=shift, angle=-angle,
                          indexing='ij')
    x, y = cc, rr  # xy frame: x increases with column, y with row

    r = np.hypot(x, y)
    rho = r/np.max(r*mask)  # rho is defined to be 1 on the edge of the aperture
    theta = np.arctan2(y, x)
    
    rho[np.where(rho==0)] = 1e-99
    theta[np.where(theta==0)] = 1e-99

    return rho, theta


def _zernike_coordinates_legacy(mask):
    # Reproduces zernike_coordinates(mask) from prtools 2.0.2 and earlier:
    # +y toward decreasing row (theta is measured toward increasing row and
    # paired with sin(m*theta) for m < 0), and the origin measured from
    # shape/2 rather than the mesh origin at floor(shape/2)
    mask = np.asarray(mask, dtype=bool)
    center = np.asarray(mask.shape)/2
    centroid = prtools.centroid(mask)
    shift = (centroid[0]-center[0], centroid[1]-center[1])
    rr, cc = prtools.mesh(shape=mask.shape, shift=shift, indexing='ij')
    r = np.abs(cc + 1j*rr)
    rho = r/np.max(r*mask)
    theta = np.angle(cc + 1j*rr)
    rho[np.where(rho==0)] = 1e-99
    theta[np.where(theta==0)] = 1e-99
    return rho, theta


def zernike_coeff_legacy(coeffs, modes=None, order='noll'):
    """Convert Zernike coefficients between legacy and current
    orientation conventions.

    prtools 2.0.2 and earlier computed Zernike polynomials in a coordinate
    system mirrored top-to-bottom (y -> -y) relative to the current one. This
    results in modes with negative azimuthal order (sine terms: y tilt, 45
    degree astigmatism, y coma, ...) having a flipped sign. Modes with zero
    or positive azimuthal order are unaffected.

    This conversion is involutory so it converts legacy coefficients to the
    current convention and current coefficients to the legacy convention.

    Parameters
    ----------
    coeffs : array_like
        Zernike coefficients. The last axis indexes Zernike modes, so a
        stack of coefficient vectors (e.g. a time series) can be converted
        in a single call.
    modes : array_like, optional
        Zernike mode indices corresponding to the last axis of ``coeffs``,
        e.g. the ``modes`` passed to :func:`zernike_fit`. If None
        (default), ``coeffs[..., k]`` is taken to be the coefficient for
        mode index ``k + 1``, matching :func:`zernike_compose`.
    order : {'noll', 'fringe', 'ansi'}, optional
        Zernike ordering scheme. Default is 'noll'.

    Returns
    -------
    ndarray
        Converted coefficients with the same shape as ``coeffs``.

    Notes
    -----
    The conversion is exact for masks with an even number of pixels in each
    dimension. Along any dimension with an odd number of pixels, the legacy
    coordinate origin was also offset half a pixel from the mask centroid.
    The effect of that offset on fitted coefficients is small but is not a
    sign change, so it is not undone by this function.

    Examples
    --------
    Convert tip, tilt, and focus coefficients fit with prtools 2.0.2:

    .. code:: pycon

        >>> prtools.zernike_coeff_legacy([1e-8, 2e-8, 3e-8], modes=[2, 3, 4])
        array([ 1.e-08, -2.e-08,  3.e-08])

    Convert a full coefficient vector as used by :func:`zernike_compose`:

    .. code:: pycon

        >>> coeffs = prtools.zernike_coeff_legacy(legacy_coeffs)
        >>> opd = prtools.zernike_compose(mask, coeffs)

    See Also
    --------
    :func:`zernike_compose`
    :func:`zernike_fit`

    """
    coeffs = np.asarray(coeffs)
    if coeffs.ndim == 0:
        raise ValueError('coeffs must have at least one dimension')

    if order == 'noll':
        to_nm = noll_to_nm
    elif order == 'fringe':
        to_nm = fringe_to_nm
    elif order == 'ansi':
        to_nm = ansi_to_nm
    else:
        raise ValueError('Unknown order', order)

    if modes is None:
        modes = np.arange(1, coeffs.shape[-1] + 1)
    else:
        modes = np.atleast_1d(np.asarray(modes))
        if modes.ndim != 1 or modes.shape[0] != coeffs.shape[-1]:
            raise ValueError(
                f'modes has {modes.size} entries but the last axis of '
                f'coeffs has length {coeffs.shape[-1]}')

    sign = np.array([-1 if to_nm(int(j))[1] < 0 else 1 for j in modes])

    return coeffs * sign
