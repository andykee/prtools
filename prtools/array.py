import warnings

from array_api_compat import is_jax_namespace as is_jax
import numpy as np

from prtools._array_api import array_namespace, scipy_namespace


def centroid(x, where=None, kind='absolute', indexing='ij'):
    """Compute array centroid location.

    Parameters
    ----------
    x : array_like
        Input array.
    where: array_like of bool, optional
        Elements to include in the centroid calculation. If None (default),
        all finite and non-NaN values are used.
    kind : {'absolute', 'center'}, optional
        Specifies the kind of centroid as a string. If 'absolute' (default),
        the absolute centroid within the input is returned. If 'center', the
        centroid relative to the center of the input is returned.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of mesh.

    Returns
    -------
    centroid : tuple
        (r, c) or (x, y) centroid location

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> circ = prtools.circle(shape=(256, 256), radius=25, shift=(-80, 50))
        >>> plt.imshow(circ, cmap='gray')

    .. code:: pycon

        >>> prtools.centroid(circ)
        (48.00000000000002, 178.0)
        >>> prtools.centroid(circ, kind='center')
        (-79.99999999999997, 50.0)
        >>> prtools.centroid(circ, kind='center', indexing='xy')
        (50.0, 79.99999999999997)

    """
    xp = array_namespace(x)
    x = xp.asarray(x)

    if kind not in ('absolute', 'center'):
        raise ValueError(f'Unknown kind {kind}')

    if indexing not in ('ij', 'xy'):
        raise ValueError("Valid values for indexing are 'xy' and 'ij'.")

    if where is None:
        where = xp.isfinite(x)
    else:
        where = xp.asarray(where, dtype=bool)

    if np.isnan(x[where]).any():
        warnings.warn('Unmasked NaN in input', RuntimeWarning,
                      stacklevel=2)

    anorm = x[where]/xp.sum(x[where])

    nr, nc = x.shape
    rr, cc = xp.mgrid[0:nr, 0:nc]

    r = xp.dot(rr[where].ravel(), anorm.ravel())
    c = xp.dot(cc[where].ravel(), anorm.ravel())

    if kind == 'center':
        rc, cc = np.array(x.shape)/2
        r = r - rc
        c = c - cc

    if indexing == 'xy':
        r, c = c, -r

    return r, c


def pad(x, shape, fill=0):
    """Zero-pad an array.

    Note that pad works accepts both two and three dimensional arrays.

    Parameters
    ----------
    x : array_like
        Array to be padded.
    shape : array_like of ints
        Shape of output array in ``(nrows, ncols)``.
    fill : scalar
        Fill vlue used when pad operation increases array size.

    Returns
    -------
    padded_array : ndarray
        Zero-padded array with shape ``(nrows, ncols)``. If ``x`` has a
        third dimension, the return shape will be ``(depth, nrows, ncols)``.

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> circ = prtools.circle(shape=(128, 128), radius=64)
        >>> circ_pad = prtools.pad(circ, shape=(200, 200))
        >>> fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 2))
        >>> ax[0].imshow(circ, cmap='gray')
        >>> ax[0].set_title('Original array')
        >>> ax[1].imshow(circ_pad, cmap='gray')
        >>> ax[1].set_title('Padded array')

    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> circ = prtools.circle(shape=(128, 128), radius=64)
        >>> circ_pad = prtools.pad(circ, shape=(110, 110))
        >>> fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5, 2))
        >>> ax[0].imshow(circ, cmap='gray')
        >>> ax[0].set_title('Original array')
        >>> ax[1].imshow(circ_pad, cmap='gray')
        >>> ax[1].set_title('Padded array')

    """
    xp = array_namespace(x)
    x = xp.atleast_2d(x)
    shape = np.broadcast_to(np.asarray(shape, dtype=int), (2,))

    if x.ndim == 2:
        out_shape = (1, int(shape[0]), int(shape[1]))
        x = x[np.newaxis, :]
    else:  # a.ndim == 3
        out_shape = (x.shape[0], int(shape[0]), int(shape[1]))


    # The row and col indices here are 1 and 2 respectively since we've
    # forced both a and out to have a 3rd dimension
    rmin = min(x.shape[1] // 2, out_shape[1] // 2)
    rmax = min(x.shape[1] - x.shape[1] // 2, out_shape[1] - out_shape[1] // 2)
    cmin = min(x.shape[2] // 2, out_shape[2] // 2)
    cmax = min(x.shape[2] - x.shape[2] // 2, out_shape[2] - out_shape[2] // 2)

    # Extract the overlapping region from the source array
    a_slc_r = slice(x.shape[1] // 2 - rmin, x.shape[1] // 2 + rmax)
    a_slc_c = slice(x.shape[2] // 2 - cmin, x.shape[2] // 2 + cmax)
    cropped = x[:, a_slc_r, a_slc_c]

    # Pad the cropped region to the output shape
    pad_r_before = out_shape[1] // 2 - rmin
    pad_r_after = out_shape[1] - out_shape[1] // 2 - rmax
    pad_c_before = out_shape[2] // 2 - cmin
    pad_c_after = out_shape[2] - out_shape[2] // 2 - cmax

    out = xp.pad(cropped,
                 ((0, 0), (pad_r_before, pad_r_after), (pad_c_before, pad_c_after)),
                 mode='constant', constant_values=fill)

    return xp.squeeze(out)


def subarray(x, shape, shift=(0, 0)):
    """Extract a contiguous subarray from a larger array.

    The subarray is extracted about the center of the source array unless
    a shift is specified.

    Parameters
    ----------
    x : array_like
        Source array
    shape : array_like of ints
        Shape of subarray array in ``(nrows, ncols)``.
    shift : array_like of ints
        Relative shift of the center of the subarray in ``(row, col)``.

    Returns
    -------
    out : ndarray
        Subarray extracted from the source array.

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> circ = prtools.circle(shape=(128,128), radius=64)
        >>> circ_subarray = prtools.subarray(circ, shape=(110,110))
        >>> fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,2))
        >>> ax[0].imshow(circ, cmap='gray')
        >>> ax[0].set_title('Original array')
        >>> ax[1].imshow(circ_subarray, cmap='gray')
        >>> ax[1].set_title('Subarray')

    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> circ = prtools.circle(shape=(128,128), radius=64)
        >>> circ_subarray = prtools.subarray(circ, shape=(64,64), shift=(32,32))
        >>> fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,2))
        >>> ax[0].imshow(circ, cmap='gray')
        >>> ax[0].set_title('Original array')
        >>> ax[1].imshow(circ_subarray, cmap='gray')
        >>> ax[1].set_title('Subarray')
    """
    xp = array_namespace(x)
    x = xp.asarray(x)
    shape = np.asarray(shape)

    rmin = x.shape[0]//2 - shape[0]//2 + shift[0]
    cmin = x.shape[1]//2 - shape[1]//2 + shift[1]
    rmax = rmin + shape[0]
    cmax = cmin + shape[1]

    if any((rmin<0, cmin<0, rmax>x.shape[0], cmax>x.shape[1])):
        raise ValueError('window lies outside of array')

    return x[rmin:rmax, cmin:cmax]


def boundary(x, threshold=0):
    """Find bounding row and column indices of data within an array.

    Parameters
    ----------
    x : array_like
        Input array

    threshold : float, optional
        Masking threshold to apply before boundary finding. Only values
        in x that are larger than threshold are considered in the boundary
        finding operation. Default is 0.

    Returns
    -------
    rmin, rmax, cmin, cmax : ints
        Boundary indices

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> circ = prtools.circle(shape=(200, 200), radius=50)
        >>> plt.imshow(circ, cmap='gray')
        >>> plt.grid('on')

    .. code:: pycon

        >>> prtools.boundary(circ)
        (50, 150, 50, 150)
    """
    xp = array_namespace(x)
    x = xp.asarray(x)
    x = (x > threshold)

    rows = xp.any(x, axis=1)
    cols = xp.any(x, axis=0)

    idx = np.array((0, -1))  # https://github.com/andykee/prtools/issues/6
    rmin, rmax = xp.where(rows)[0][idx]
    cmin, cmax = xp.where(cols)[0][idx]

    return rmin, rmax, cmin, cmax


def rebin(x, factor):
    """Rebin an array by an integer factor.

    Parameters
    ----------
    x : array_like
        Array or cube of arrays to rebin. If a cube is provided, the first
        dimension should index the image slices.

    factor : int
        Rebinning factor

    Returns
    -------
    x : ndarray
        Rebinned array

    See Also
    --------
    :func:`rescale`

    """
    xp = array_namespace(x)
    x = xp.asarray(x)

    if xp.iscomplexobj(x):
        raise ValueError('rebin is not defined for complex data')

    if x.ndim == 3:
        if is_jax(xp):
            import jax
            fun = lambda a, f: xp.reshape(a, (a.shape[0]//f, f, a.shape[1]//f, f)).sum(-1).sum(1)
            xr = jax.vmap(fun, in_axes=[0, None])(x, factor)
        else:
            rebinned_shape = (x.shape[0], x.shape[1]//factor, x.shape[2]//factor)
            xr = np.zeros(rebinned_shape, dtype=x.dtype)
            for i in range(x.shape[0]):
                xr[i] = x[i].reshape(rebinned_shape[1], factor,
                                     rebinned_shape[2], factor).sum(-1).sum(1)
    else:
        xr = x.reshape(x.shape[0]//factor, factor, 
                       x.shape[1]//factor, factor).sum(-1).sum(1)

    return xr


def rescale(img, scale, shape=None, mask=None, order=3, mode='nearest',
            unitary=True):
    """Rescale an image by interpolation.

    Parameters
    ----------
    img : array_like
        Image to rescale

    scale : float or tuple of floats
        Scaling factor. If scale is a tuple, img is rescaled according to
        ``scale_row, scale_col)``. A single value rescales img equally in both
        dimensions. Scale factors less than 1 will shrink the image. Scale
        factors greater than 1 will grow the image.

    shape : array_like or int, optional
        Output shape. If None (default), the output shape will be the input img
        shape multiplied by the scale factor.

    mask : array_like, optional
        Binary mask applied after rescaling. If None (default), a mask is
        created from the nonzero portions of img. To skip masking operation,
        set ``mask = np.ones_like(img)``

    order : int, optional
        Order of spline interpolation used for rescaling operation. Default is
        3. Order must be in the range 0-5.

    mode : {'constant', 'nearest', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according to the
        given mode. Default is 'nearest'.

    unitary : bool, optional
        Normalization flag. If True (default), a normalization is performed on
        the output such that the rescaling operation is unitary and image power
        (if complex) or intensity (if real) is conserved.

    Returns
    -------
    ndarray

    Note
    ----
    The post-rescale masking operation should have no real effect on the
    resulting image but is included to eliminate interpolation artifacts that
    sometimes appear in large clusters of zeros in rescaled images.

    See Also
    --------
    :func:`rebin`

    """
    xp = array_namespace(img)
    scipy = scipy_namespace(xp)

    img = xp.asarray(img)
    scale = np.broadcast_to(scale, (2,))

    if shape is None:
        shape = img.shape
    shape = np.broadcast_to(shape, (2,))
    shape = np.ceil(shape * scale).astype(int)

    if mask is None:
        # take the real portion to ensure that even if img is complex, mask
        # will be real
        mask = np.zeros_like(img).real
        mask[img != 0] = 1
        mask = xp.asarray(mask)

    r = (np.arange(shape[0], dtype=np.float64) - shape[0]/2.)/scale[0] + img.shape[0]/2.
    c = (np.arange(shape[1], dtype=np.float64) - shape[1]/2.)/scale[1] + img.shape[1]/2.

    rr, cc = np.meshgrid(r, c, indexing='ij')

    mask = scipy.ndimage.map_coordinates(mask, [rr, cc], order=1, mode='nearest')
    mask[mask < np.finfo(mask.dtype).eps] = 0

    if np.iscomplexobj(img):
        out_real = scipy.ndimage.map_coordinates(img.real, [rr, cc], order=order, mode=mode)
        out_imag = scipy.ndimage.map_coordinates(img.imag, [rr, cc], order=order, mode=mode)
        out = out_real + 1j*out_imag
    else:
        out = scipy.ndimage.map_coordinates(img, [rr, cc], order=order, mode=mode)

    if unitary:
        out = out * xp.sum(img)/xp.sum(out)

    out = out * mask

    return out


def normpow(x, power=1):
    r"""Normalizie the power in an array.

    The total power in an array is given by

    .. math::

        P = \sum{\left|\mbox{array}\right|^2}

    A normalization coefficient is computed as

    .. math::

        c = \sqrt{\frac{p}{\sum{\left|\mbox{array}\right|^2}}}

    The array returned will be scaled by the normalization coefficient so
    that its power is equal to :math:`p`.

    Parameters
    ----------
    x : array_like
        Array to be normalized

    power : float, optional
        Desired power in normalized array. Default is 1.

    Returns
    -------
    x : ndarray
        Normalized array

    """
    xp = array_namespace(x)
    x = xp.asarray(x)
    return x * xp.sqrt(power/xp.sum(xp.abs(x)**2))


def shift(x, shift, mode='wrap', fill=0.0):
    """Shift an array via FFT.

    Shift an array by (row, column). The shifts may be non-integer as the
    shift operation is implemented by introducing a Fourier-domain tilt. If
    ``a`` is complex, the result will also be complex.

    Parameters
    ----------
    x : array_like
        The input array.
    shift : (2,) sequence
        The shift specified as (row, column).
    mode : {'wrap', 'reflect', 'mirror', 'constant'}, optional
        Determines how the input array is extended beyond its boundaries.
        Default is 'wrap'.

        * 'wrap' (a b c d | a b c d | a b c d)
            The input is extended by wrapping around to the opposite edge.
        * 'reflect' (d c b a | a b c d | d c b a)
            The input is extended by reflecting about the edge of the last
            pixel. This mode is also sometimes referred to as half-sample
            symmetric.
        * 'mirror' (d c b | a b c d | c b a)
            The input is extended by reflecting about the center of the last
            pixel. This mode is also sometimes referred to as whole-sample
            symmetric.
        * 'constant'
            The input is extended by filling all values beyond the edge with
            the same constant value, defined by the fill parameter.

    fill : scalar, optional
        Value to fill past edges if `mode` is 'constant'. Default is 0.0
    Returns
    -------
    x : ndarray
        The shifted input array.

    Example
    -------
    .. code:: pycon

        >>> arr = np.zeros((3,3))
        >>> arr[2,2] = 1
        >>> arr
        array([[0., 0., 0.],
               [0., 0., 0.],
               [0., 0., 1.]])
        >>> arr_shift = prtools.shift(arr, shift=(-1,-1))
        >>> arr_shift
        array([[ 0.00000000e+00, -7.40148683e-17, -2.46716228e-17],
               [-1.16747372e-16,  1.00000000e+00,  2.14548192e-16],
               [-3.12823642e-17,  2.22044605e-16, -4.18468327e-17]])
    """
    xp = array_namespace(x)
    x = xp.asarray(x)
    r, c = x.shape
    dr, dc = shift

    x = _extend(x, shift, mode, fill, xp=xp)

    R = dr * xp.fft.fftfreq(x.shape[0])
    C = dc * xp.fft.fftfreq(x.shape[1])

    RR, CC = xp.meshgrid(R, C, indexing='ij')
    K = xp.exp(-1j*2*xp.pi*(RR+CC))
    shifted = xp.fft.ifft2(xp.fft.fft2(x)*K)

    shifted = pad(shifted, (r, c))  # crop back to original size

    if xp.any(xp.iscomplex(x)):
        return shifted
    else:
        return shifted.real


def _extend(a, shift, mode, fill, xp):
    r, c = a.shape
    dr, dc = np.ceil(np.abs(shift)).astype(int)

    if mode == 'wrap':
        return a

    if mode == 'constant':
        return pad(a, shape=(r + 2*dr, c + 2*dc), fill=fill)

    if mode in ('reflect', 'mirror'):
        # mirror excludes the boundary element; reflect includes it
        o = 1 if mode == 'mirror' else 0

        # Source slices for the border regions
        tr = slice(o, dr + o)              # top rows
        br = slice(r - dr - o, r - o)      # bottom rows
        lc = slice(o, dc + o)              # left cols
        rc = slice(c - dc - o, c - o)      # right cols

        # Build the 3x3 grid of regions and concatenate
        top_row = xp.concat([
            xp.flip(a[tr, lc]),                 # upper-left corner
            xp.flip(a[tr, :], axis=0),          # top edge
            xp.flip(a[tr, rc]),                 # upper-right corner
        ], axis=1)

        mid_row = xp.concat([
            xp.flip(a[:, lc], axis=1),          # left edge
            a,                                   # center
            xp.flip(a[:, rc], axis=1),          # right edge
        ], axis=1)

        bot_row = xp.concat([
            xp.flip(a[br, lc]),                 # lower-left corner
            xp.flip(a[br, :], axis=0),          # bottom edge
            xp.flip(a[br, rc]),                 # lower-right corner
        ], axis=1)

        return xp.concat([top_row, mid_row, bot_row], axis=0)

    raise ValueError(f'Unknown mode {mode}')


def register(x1, x2, oversample, return_error=False):
    """Compute the subpixel image translation to register the input array 
    ``x1`` to a reference array ``x2``.

    The registration shift is computed in two steps: first a coarse estimate
    is computed from the FFT-based cross-correlation of the two input arrays.
    This estimate is then refined to subpixel accuracy by computing the
    upsampled DFT-based cross-correlation in a small neigborhood around the
    initial estimate.

    Parameters
    ----------
    x1 : array_like
        Array to register.
    x2 : array_like
        Target array.
    oversample : float
        Oversampling factor for subpixel registration. Registration accuracy
        is 1/oversample.
    return_error : bool, optional
        If True, the noramlized RMS registration error is returned. Default is
        False.
    Returns
    -------
    shift : tuple
        Translation in (row, col) that will register *arr* to *ref*.
    err : float
        Registration error

    References
    ----------
    Guizar-Sicairos, Thurman, and Fienup, "Efficient subpixel image
    registration algorithms". Optics Letters 33, 156-158 (2008)

    See also
    --------
    :func:`~shift`

    Example
    -------
    .. code:: pycon

        >>> ref = np.zeros((3,3))
        >>> ref[1,1] = 1
        >>> arr = np.zeros((3,3))
        >>> arr[2,2] = 1
        >>> shift = prtools.register(arr, ref, oversample=2)
        >>> shift
        (-1.0, -1.0)

    """
    xp = array_namespace(x1, x2)
    x1 = xp.fft.fft2(x1)
    x2 = xp.fft.fft2(x2)
    xcorr = xp.fft.fftshift(xp.fft.ifft2(x2*xp.conj(x1)))

    # find peak
    maxima = xp.asarray(xp.unravel_index(xp.argmax(xp.abs(xcorr)), xcorr.shape))
    peak = xcorr[maxima]

    # compute shifts
    center = xp.array([xp.fix(x/2) for x in x1.shape])
    shift = maxima - center
    if oversample != 1:
        # now we can set up and perform the oversampled dft on an oversampled
        # 1.5 x 1.5 pixel region about the peak
        npix_dft = xp.ceil(oversample*1.5)
        dft_shift = xp.fix(npix_dft/2)
        rs = dft_shift - shift[0] * oversample
        cs = dft_shift - shift[1] * oversample

        # Compute DFT
        X = xp.arange(x1.shape[1]) - xp.floor(x1.shape[1]/2)
        Y = xp.arange(x1.shape[0]) - xp.floor(x1.shape[0]/2)
        U = xp.arange(npix_dft) - cs
        V = xp.arange(npix_dft) - rs
        E1 = xp.exp(-2*xp.pi*1j/(x1.shape[0]*oversample)*xp.outer(V, Y))
        E2 = xp.exp(-2*xp.pi*1j/(x1.shape[1]*oversample)*xp.outer(X, U))
        xcorr = xp.dot(xp.dot(E1, xp.conj(xp.fft.ifftshift(x2*xp.conj(x1)))), E2)

        maxima_subpx = xp.asarray(xp.unravel_index(xp.argmax(xp.abs(xcorr)), xcorr.shape))
        peak = xcorr[maxima_subpx]

        # Combine subpixel peak coordinates with integer pixel peak coords
        maxima_subpx = maxima_subpx - dft_shift
        shift = shift + maxima_subpx/oversample

    shift = tuple(shift)

    # Compute normalized RMS error
    if return_error:
        x1_amp = xp.sum(xp.abs(x1)**2)
        ref_amp = xp.sum(xp.abs(x2)**2)
        err = 1-xp.abs(peak)**2/(x1_amp*ref_amp)
        err = xp.sqrt(xp.abs(err))
        return shift, err

    return shift


def medfix(x, mask, kernel=(3, 3), nanwarn=False):
    """Fix masked entries in a 2-dimensional array via median filtering.

    Parameters
    ----------
    input : array_like
        A 2-dimensional input array
    mask : array_like
        A 2-dimensional mask with the same shape as input. Entries which
        evaluate to True are considered masked and will be repaired.
    kernel : array_like, optional
        A scalar or list of length 2 specifying the filter window in
        each dimension. Elements of *kernel* must be odd. If *kernel*
        is a scalar, it is used for each dimension. Default is (3,3).
    nanwarn : bool, optional
        If True, a RuntimeWarning will be raised if NaNs are present
        in the output. Default is False.

    Returns
    -------
    ndarray

    Notes
    -----
    Masked areas larger than the kernel size will introduce NaNs into
    the output.

    """
    xp = array_namespace(x)
    x = xp.array(x, dtype=float, copy=True)  # force a copy
    mask = xp.asarray(mask, dtype=bool)

    if not mask.any():
        # nothing to do
        return x

    kernel = xp.asarray(kernel)
    if kernel.shape == ():
        kernel = xp.repeat(kernel, 2)
    if np.any(kernel % 2 == 0):
        raise ValueError("Kernel must be odd sized")

    if is_jax(xp):
        x = x.at[mask].set(xp.nan)
    else:
        x[mask] = xp.nan

    pw = (kernel - 1)//2
    pad_width = ((pw[0], pw[0]), (pw[1], pw[1]))

    x_pad = xp.pad(x, pad_width=pad_width, mode='constant',
                       constant_values=xp.nan)

    i, j = xp.nonzero(mask)  # indices of bad pixels
    i_pad, j_pad = i + pw[0], j + pw[1]  # indices offset by padding width

    # define neighborhood offsets
    di = xp.arange(kernel[0]) - pw[0]
    dj = xp.arange(kernel[1]) - pw[1]
    window = xp.stack(xp.meshgrid(di, dj, indexing='ij'), axis=-1).reshape(-1, 2)  # shape (prod(kernel), 2)

    # compute all neighborhood coordinates
    rows = i_pad[:, None] + window[:, 0]  # shape (num_bad_px, prod(kernel))
    cols = j_pad[:, None] + window[:, 1]  # shape (num_bad_px, prod(kernel))

    # extract neighborhoods using advanced indexing and compute replacement
    # values
    bad_px_kernel = x_pad[rows, cols]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        bad_px_vals = xp.nanmedian(bad_px_kernel, axis=1)

    if is_jax(xp):
        x = x.at[i, j].set(bad_px_vals)
    else:
        x[i, j] = bad_px_vals

    if nanwarn and xp.isnan(x).any():
        warnings.warn('Result contains NaNs', RuntimeWarning,
                      stacklevel=2)

    return x
