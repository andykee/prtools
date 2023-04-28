import numpy as np
from numpy.lib.stride_tricks import as_strided
from scipy.ndimage import map_coordinates


def centroid(a):
    """Compute array centroid location.

    Parameters
    ----------
    a : array_like
        Input array.

    Returns
    -------
    centroid : tuple
        (r, c) centroid location.

    """
    a = np.asarray(a)
    a = a/np.sum(a)
    nr, nc = a.shape
    rr, cc = np.mgrid[0:nr, 0:nc]

    r = np.dot(rr.ravel(), a.ravel())
    c = np.dot(cc.ravel(), a.ravel())

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
    """

    a = np.atleast_2d(a)
    shape = np.broadcast_to(np.asarray(shape, dtype=int), (2,))    

    if a.ndim == 2:
        shape = np.append(1, shape)
        a = a[np.newaxis,:]         
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
        Array or cube of arrays to rebin. If a cube is provided, the first dimension
        should index the image slices.

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

    scale : float
        Scaling factor. Scale factors less than 1 will shrink the image. Scale
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
        given mode. Default is 'constant'.

    unitary : bool, optional
        Normalization flag. If True (default), a normalization is performed on
        the output such that the rescaling operation is unitary and image power
        (if complex) or intensity (if real) is conserved.

    Returns
    -------
    img : ndarray
        Rescaled image.

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

    if mask is None:
        # take the real portion to ensure that even if img is complex, mask will
        # be real
        mask = np.zeros_like(img).real
        mask[img != 0] = 1

    if shape is None:
        shape = np.ceil((img.shape[0]*scale, img.shape[1]*scale)).astype(int)
    else:
        if np.isscalar(shape):
            shape = np.ceil((shape*scale, shape*scale)).astype(int)
        else:
            shape = np.ceil((shape[0]*scale, shape[1]*scale)).astype(int)

    x = (np.arange(shape[1], dtype=np.float64) - shape[1]/2.)/scale + img.shape[1]/2.
    y = (np.arange(shape[0], dtype=np.float64) - shape[0]/2.)/scale + img.shape[0]/2.

    xx, yy = np.meshgrid(x, y)

    mask = map_coordinates(mask, [yy, xx], order=1, mode='nearest')
    mask[mask < np.finfo(mask.dtype).eps] = 0

    if np.iscomplexobj(img):
        out = np.zeros(shape, dtype=np.complex128)
        out.real = map_coordinates(img.real, [yy, xx], order=order, mode=mode)
        out.imag = map_coordinates(img.imag, [yy, xx], order=order, mode=mode)
    else:
        out = map_coordinates(img, [yy, xx], order=order, mode=mode)

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



def shift(a, shift):
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
    Returns
    -------
    shifted : ndarray
        The shifted input array.

    Example
    -------
    .. code:: pycon

        >>> import prtools
        >>> import numpy as np
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

    dr, dc = shift
    R = dr * np.fft.fftfreq(a.shape[0])
    C = dc * np.fft.fftfreq(a.shape[1])

    RR, CC = np.meshgrid(R, C, indexing='ij')
    K = np.exp(-1j*2*np.pi*(RR+CC))
    shifted = np.fft.ifft2(np.fft.fft2(a)*K)

    if np.any(np.iscomplex(a)):
        return shifted
    else:
        return shifted.real
    

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

        >>> import prtools
        >>> import numpy as np
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
    if oversample !=1:
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
        E1 = np.exp(-2*np.pi*1j/(arr.shape[0]*oversample)*np.outer(V,Y))
        E2 = np.exp(-2*np.pi*1j/(arr.shape[1]*oversample)*np.outer(X,U))
        xcorr = np.dot(np.dot(E1,np.conj(np.fft.ifftshift(G*np.conj(F)))),E2)
        
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


def medfix2(input, mask, kernel=(3,3)):
    """Fix masked entries in a 2-dimensional array via median filtering.

    Parameters
    ----------
    input : array_like
        A 2-dimensional input array
    mask : array_like
        A 2-dimensional
    kernel : array_like, optional
        A scalar or list of length 2 specifying the filter window in 
        each dimension. Elements of *kernel* should be odd. If *kernel*
        is a scalar, it is used for each dimension. Default is (3,3).
    Returns
    -------
    out : ndarray
        An array the same size as input where the masked entries are 
        replaced with median filtered values.
    
    """
    # force a copy by calling array instead of asarray
    input = np.array(input, dtype=float)

    mask = np.asarray(mask, dtype=bool)

    kernel = np.asarray(kernel)
    if kernel.shape == ():
        kernel = np.repeat(kernel, 2)
    if np.any(kernel % 2 == 0):
        raise ValueError("Each element of kernel must be odd")

    input[mask] = np.nan

    pad_size = np.array((kernel-1)/2, dtype=int)
    input_pad = np.full((input.shape[0] + pad_size[0] * 2, 
                         input.shape[1] + pad_size[1] * 2), np.nan)
    slc = (slice(pad_size[0], pad_size[0] + input.shape[0]), 
           slice(pad_size[1], pad_size[1] + input.shape[1]))
    input_pad[slc] = input

    input_pad = as_strided(input_pad, 
                           shape=(input_pad.shape[0], input_pad.shape[1], kernel[0], kernel[1]),
                           strides=input_pad.strides + input_pad.strides)
    input_pad = input_pad.reshape((input_pad.shape[0], input_pad.shape[1], np.prod(kernel)))

    mask_idx = np.where(mask)

    input[mask_idx] = np.nanmedian(input_pad[mask_idx[0], mask_idx[1], :], axis=1)

    return input