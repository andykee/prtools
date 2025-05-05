import warnings

from prtools import __backend__
from prtools._backend import numpy as np
from prtools._backend import scipy


def centroid(a, where=None, kind='absolute', indexing='ij'):
    """Compute array centroid location.

    Parameters
    ----------
    a : array_like
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
    if kind not in ('absolute', 'center'):
        raise ValueError(f'Unknown kind {kind}')

    if indexing not in ('ij', 'xy'):
        raise ValueError("Valid values for indexing are 'xy' and 'ij'.")

    a = np.asarray(a)

    if where is None:
        where = np.isfinite(a)
    else:
        where = np.asarray(where, dtype=bool)

    if np.isnan(a[where]).any():
        warnings.warn('Unmasked NaN in input', RuntimeWarning,
                      stacklevel=2)

    anorm = a[where]/np.sum(a[where])

    nr, nc = a.shape
    rr, cc = np.mgrid[0:nr, 0:nc]

    r = np.dot(rr[where].ravel(), anorm.ravel())
    c = np.dot(cc[where].ravel(), anorm.ravel())

    if kind == 'center':
        rc, cc = np.array(a.shape)/2
        r -= rc
        c -= cc

    if indexing == 'xy':
        r, c = c, -r

    return r, c


def pad(a, shape, fill=0):
    """Zero-pad an array.

    Note that pad works accepts both two and three dimensional arrays.

    Parameters
    ----------
    a : array_like
        Array to be padded.
    shape : array_like of ints
        Shape of output array in ``(nrows, ncols)``.
    fill : scalar
        Fill vlue used when pad operation increases array size.

    Returns
    -------
    padded_array : ndarray
        Zero-padded array with shape ``(nrows, ncols)``. If ``a`` has a
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
    a = np.atleast_2d(a)
    shape = np.broadcast_to(np.asarray(shape, dtype=int), (2,))

    if a.ndim == 2:
        shape = np.append(1, shape)
        a = a[np.newaxis, :]
    else:  # a.ndim == 3
        shape = np.append(a.shape[0], shape)

    out = np.ones(shape, dtype=a.dtype) * fill

    # The row and col indices here are 1 and 2 respectively since we've
    # forced both a and out to have a 3rd dimension
    rmin = np.min([a.shape[1]//2, out.shape[1]//2])
    rmax = np.min([a.shape[1]-a.shape[1]//2, out.shape[1]-out.shape[1]//2])
    cmin = np.min([a.shape[2]//2, out.shape[2]//2])
    cmax = np.min([a.shape[2]-a.shape[2]//2, out.shape[2]-out.shape[2]//2])

    a_slc_r = slice(a.shape[1]//2-rmin, a.shape[1]//2+rmax)
    a_slc_c = slice(a.shape[2]//2-cmin, a.shape[2]//2+cmax)

    out_slc_r = slice(out.shape[1]//2-rmin, out.shape[1]//2+rmax)
    out_slc_c = slice(out.shape[2]//2-cmin, out.shape[2]//2+cmax)

    out[:, out_slc_r, out_slc_c] = a[:, a_slc_r, a_slc_c]
    return np.squeeze(out)


def subarray(a, shape, shift=(0, 0)):
    """Extract a contiguous subarray from a larger array.

    The subarray is extracted about the center of the source array unless
    a shift is specified.

    Parameters
    ----------
    a : array_like
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

    a = np.asarray(a)
    shape = np.asarray(shape)

    rmin = a.shape[0]//2 - shape[0]//2 + shift[0]
    cmin = a.shape[1]//2 - shape[1]//2 + shift[1]
    rmax = rmin + shape[0]
    cmax = cmin + shape[1]

    if any((rmin<0, cmin<0, rmax>a.shape[0], cmax>a.shape[1])):
        raise ValueError('window lies outside of array')

    return a[rmin:rmax, cmin:cmax]


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
    x = np.asarray(x)
    x = (x > threshold)

    rows = np.any(x, axis=1)
    cols = np.any(x, axis=0)

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax


def rebin(img, factor):
    """Rebin an image by an integer factor.

    Parameters
    ----------
    img : array_like
        Array or cube of arrays to rebin. If a cube is provided, the first
        dimension should index the image slices.

    factor : int
        Rebinning factor

    Returns
    -------
    img : ndarray
        Rebinned image

    See Also
    --------
    :func:`rescale`

    """
    img = np.asarray(img)

    if np.iscomplexobj(img):
        raise ValueError('rebin is not defined for complex data')

    if img.ndim == 3:
        rebinned_shape = (img.shape[0], img.shape[1]//factor, img.shape[2]//factor)
        img_rebinned = np.zeros(rebinned_shape, dtype=img.dtype)
        for i in range(img.shape[0]):
            img_rebinned[i] = img[i].reshape(rebinned_shape[1], factor,
                                             rebinned_shape[2], factor).sum(-1).sum(1)
    else:
        img_rebinned = img.reshape(img.shape[0]//factor, factor, img.shape[1]//factor,
                                   factor).sum(-1).sum(1)

    return img_rebinned


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

    img = np.asarray(img)
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

    r = (np.arange(shape[0], dtype=np.float64) - shape[0]/2.)/scale[0] + img.shape[0]/2.
    c = (np.arange(shape[1], dtype=np.float64) - shape[1]/2.)/scale[1] + img.shape[1]/2.

    rr, cc = np.meshgrid(r, c, indexing='ij')

    mask = scipy.ndimage.map_coordinates(mask, [rr, cc], order=1, mode='nearest')
    mask[mask < np.finfo(mask.dtype).eps] = 0

    if np.iscomplexobj(img):
        out = np.zeros(shape, dtype=np.complex128)
        out.real = scipy.ndimage.map_coordinates(img.real, [rr, cc], order=order, mode=mode)
        out.imag = scipy.ndimage.map_coordinates(img.imag, [rr, cc], order=order, mode=mode)
    else:
        out = scipy.ndimage.map_coordinates(img, [rr, cc], order=order, mode=mode)

    if unitary:
        out *= np.sum(img)/np.sum(out)

    out *= mask

    return out


def normpow(array, power=1):
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
    array : array_like
        Array to be normalized

    power : float, optional
        Desired power in normalized array. Default is 1.

    Returns
    -------
    array : ndarray
        Normalized array

    """
    array = np.asarray(array)
    return array * np.sqrt(power/np.sum(np.abs(array)**2))


def shift(a, shift, mode='wrap', fill=0.0):
    """Shift an array via FFT.

    Shift an array by (row, column). The shifts may be non-integer as the
    shift operation is implemented by introducing a Fourier-domain tilt. If
    ``a`` is complex, the result will also be complex.

    Parameters
    ----------
    a : array_like
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
    shifted : ndarray
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
    a = np.asarray(a)
    r, c = a.shape
    dr, dc = shift

    a = _extend(a, shift, mode, fill)

    R = dr * np.fft.fftfreq(a.shape[0])
    C = dc * np.fft.fftfreq(a.shape[1])

    RR, CC = np.meshgrid(R, C, indexing='ij')
    K = np.exp(-1j*2*np.pi*(RR+CC))
    shifted = np.fft.ifft2(np.fft.fft2(a)*K)

    shifted = pad(shifted, (r, c))  # crop back to original size

    if np.any(np.iscomplex(a)):
        return shifted
    else:
        return shifted.real


def _extend(a, shift, mode, fill):
    r, c = a.shape
    dr, dc = np.ceil(np.abs(shift)).astype(int)

    if mode == 'wrap':  # (a b c d | a b c d | a b c d)
        # no need to do anything since the Fourier transform
        # already does this
        out = a
    else:
        # pad the array to accomodate for
        out = pad(a, shape=(r+2*dr, c+2*dc), fill=fill)

        if mode == 'constant':  # (k k k k | a b c d | k k k k)
            pass
        elif mode == 'reflect':  # (d c b a | a b c d | d c b a)
            out[0:dr, 0:dc] = np.flip(a[0:dr, 0:dc])  # upper left
            out[0:dr, dc:c+dc] = np.flip(a[0:dr, :], axis=0)  # top
            out[0:dr, c+dc:] = np.flip(a[0:dr, c-dc:])  # upper right
            out[dr:r+dr, 0:dc] = np.flip(a[:, 0:dc], axis=1)  # left
            out[dr:r+dr, c+dc:] = np.flip(a[:, c-dc:], axis=1)  # right
            out[r+dr:, 0:dc] = np.flip(a[r-dr:r, 0:dc])  # lower left
            out[r+dr:, dc:c+dc] = np.flip(a[r-dr:r, :], axis=0)  # bottom
            out[r+dr:, c+dc:] = np.flip(a[r-dr:r, c-dc:c])  # lower right
        elif mode == 'mirror':  # (d c b | a b c d | c b a)
            out[0:dr, 0:dc] = np.flip(a[1:dr+1, 1:dc+1])  # upper left
            out[0:dr, dc:c+dc] = np.flip(a[1:dr+1, :], axis=0)  # top
            out[0:dr, c+dc:] = np.flip(a[1:dr+1, c-dc-1:c-1])  # upper right
            out[dr:r+dr, 0:dc] = np.flip(a[:, 1:dc+1], axis=1)  # left
            out[dr:r+dr, c+dc:] = np.flip(a[:, c-dc-1:c-1], axis=1)  # right
            out[r+dr:, 0:dc] = np.flip(a[r-dr-1:r-1, 1:dc+1])  # lower left
            out[r+dr:, dc:c+dc] = np.flip(a[r-dr-1:r-1, :], axis=0)  # bottom
            out[r+dr:, c+dc:] = np.flip(a[r-dr-1:r-1, c-dc-1:c-1])  # lower right
        else:
            raise ValueError(f'Unknown mode {mode}')
    return out


def register(arr, ref, oversample, return_error=False):
    """Compute the subpixel image translation to register the input array to a
    reference array.

    The registration shift is computed in two steps: first a coarse estimate
    is computed from the FFT-based cross-correlation of the two input arrays.
    This estimate is then refined to subpixel accuracy by computing the
    upsampled DFT-based cross-correlation in a small neigborhood around the
    initial estimate.

    Parameters
    ----------
    arr : array_like
        Array to register.
    ref : array_like
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
    F = np.fft.fft2(arr)
    G = np.fft.fft2(ref)
    xcorr = np.fft.fftshift(np.fft.ifft2(G*np.conj(F)))

    # find peak
    maxima = np.unravel_index(np.argmax(np.abs(xcorr)), xcorr.shape)
    peak = xcorr[maxima]

    # compute shifts
    center = np.array([np.fix(x/2) for x in arr.shape])
    shift = maxima - center
    if oversample != 1:
        # now we can set up and perform the oversampled dft on an oversampled
        # 1.5 x 1.5 pixel region about the peak
        npix_dft = np.ceil(oversample*1.5)
        dft_shift = np.fix(npix_dft/2)
        rs = dft_shift - shift[0] * oversample
        cs = dft_shift - shift[1] * oversample

        # Compute DFT
        X = np.arange(arr.shape[1]) - np.floor(arr.shape[1]/2)
        Y = np.arange(arr.shape[0]) - np.floor(arr.shape[0]/2)
        U = np.arange(npix_dft) - cs
        V = np.arange(npix_dft) - rs
        E1 = np.exp(-2*np.pi*1j/(arr.shape[0]*oversample)*np.outer(V, Y))
        E2 = np.exp(-2*np.pi*1j/(arr.shape[1]*oversample)*np.outer(X, U))
        xcorr = np.dot(np.dot(E1, np.conj(np.fft.ifftshift(G*np.conj(F)))), E2)

        maxima_subpx = np.unravel_index(np.argmax(np.abs(xcorr)), xcorr.shape)
        peak = xcorr[maxima_subpx]

        # Combine subpixel peak coordinates with integer pixel peak coords
        maxima_subpx -= dft_shift
        shift[0] += maxima_subpx[0]/oversample
        shift[1] += maxima_subpx[1]/oversample

    shift = tuple(shift)

    # Compute normalized RMS error
    if return_error:
        arr_amp = np.sum(np.abs(F)**2)
        ref_amp = np.sum(np.abs(G)**2)
        err = 1-np.abs(peak)**2/(arr_amp*ref_amp)
        err = np.sqrt(np.abs(err))
        return shift, err

    return shift


def medfix(input, mask, kernel=(3, 3), nanwarn=False):
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
    # force a copy by calling array instead of asarray
    input = np.array(input, dtype=float)
    mask = np.asarray(mask, dtype=bool)

    if not mask.any():
        # nothing to do
        return input

    kernel = np.asarray(kernel)
    if kernel.shape == ():
        kernel = np.repeat(kernel, 2)
    if np.any(kernel % 2 == 0):
        raise ValueError("Kernel must be odd sized")

    if __backend__ == 'jax':
        input = input.at[mask].set(np.nan)
    else:
        input[mask] = np.nan

    pw = (kernel - 1)//2
    pad_width = ((pw[0], pw[0]), (pw[1], pw[1]))

    input_pad = np.pad(input, pad_width=pad_width, mode='constant',
                       constant_values=np.nan)

    i, j = np.nonzero(mask)  # indices of bad pixels
    i_pad, j_pad = i + pw[0], j + pw[1]  # indices offset by padding width

    # define neighborhood offsets
    di = np.arange(kernel[0]) - pw[0]
    dj = np.arange(kernel[1]) - pw[1]
    window = np.stack(np.meshgrid(di, dj, indexing='ij'), axis=-1).reshape(-1, 2)  # shape (prod(kernel), 2)

    # compute all neighborhood coordinates
    rows = i_pad[:, None] + window[:, 0]  # shape (num_bad_px, prod(kernel))
    cols = j_pad[:, None] + window[:, 1]  # shape (num_bad_px, prod(kernel))

    # extract neighborhoods using advanced indexing and compute replacement
    # values
    bad_px_kernel = input_pad[rows, cols]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        bad_px_vals = np.nanmedian(bad_px_kernel, axis=1)

    if __backend__ == 'jax':
        input = input.at[i, j].set(bad_px_vals)
    else:
        input[i, j] = bad_px_vals

    if nanwarn and np.isnan(input).any():
        warnings.warn('Result contains NaNs', RuntimeWarning,
                      stacklevel=2)

    return input
