from prtools.backend import numpy as np


def fftconv(array, kernel, normalize_kernel=True, fft_array=True,
            fft_kernel=True):
    r"""Convolve an array with a kernel using the FFT.

    The colvolution is computed as

    .. math::

        u*v = \mathcal{F}^{-1}\left\{\mathcal{F}\{u\}\cdot\mathcal{F}\{v\}\right\}

    Parameters
    ----------
    array : array_like
        Array to be convolved with `kernel`.
    kernel : array_like
        Convolution kernel. Should have the same shape as `array`.
    normalize_kernel : bool, optional
        If True (default), kernel is normalized so that `kernel =
        kernel / np.sum(kernel)`.
    fft_array : bool, optional
        If True (default), the array is assumed to be provided in the spatial
        domain. If False, the array is assumed to be provided in the frequency
        domain.
    fft_kernel : bool, optional
        If True (default), the kernel is assumed to be provided in the spatial
        domain. If False, the kernel is assumed to be provided in the frequency
        domain.

    Returns
    -------
    ndarray

    See also
    --------
    :func:`~prtools.gauss_blur`
    :func:`~prtools.pixelate`
    """

    a = np.fft.fft2(array) if fft_array else array
    k = np.fft.fft2(array) if fft_kernel else kernel

    if normalize_kernel:
        k /= np.sum(k)

    return np.fft.ifft2(a*k).real


def gauss_blur(img, sigma, oversample=1):
    """Blur an image using a Gaussian filter using the FFT. 

    Parameters
    ----------
    img : array_like
        Array to be blurred
    sigma : float or (2,) array_like
        Standard deviation of Gaussian. Providing two values allows for
        non-symmetric Gaussian interpreted as `(sigma_row, sigma_col)`
    oversample : float, optional
        Oversampling factor of `img`. Default is 1.
    
    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> img = prtools.circle((256,256), 100)
        >>> img_blurred = prtools.gauss_blur(img, sigma=3)
        >>> fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,2))
        >>> ax[0].imshow(img, cmap='gray')
        >>> ax[0].set_title('Input image')
        >>> ax[1].imshow(img_blurred, cmap='gray')
        >>> ax[1].set_title('Blurred image')
    """
    img = np.asarray(img)
    kernel = gauss_kernel(img.shape, sigma, oversample, fftshift=False)
    return fftconv(img, kernel, normalize_kernel=True, fft_array=True,
                   fft_kernel=False)


def pixelate(img, oversample=1):
    """Apply the aperture effects of an idealized square pixel on a discretely
    sampled image using the FFT.

    Parameters
    ----------
    img : array_like
        Input image
    oversample : float, optional
        Oversampling factor of `img`. Default is 1.

    Returns
    -------
    out : ndarray
        Image with pixel sampling effects applied.

    Notes
    -----
    To avoid the introduction of numerical artifacts, this function should be
    performed on data that is at least 2x oversampled.

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> img = prtools.circle((17,17), 5)
        >>> img_pixelate = prtools.pixelate(img)
        >>> fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(5,2))
        >>> ax[0].imshow(img, cmap='gray')
        >>> ax[0].set_title('Input image')
        >>> ax[1].imshow(img_pixelate, cmap='gray')
        >>> ax[1].set_title('Pixelated image')

    Note that the pixelation process preserves image power:

    .. code:: pycon

        >>> print(np.sum(img), np.sum(img_pixelate))
        64.0, 64.0
    """
    img = np.asarray(img)
    kernel = pixel_kernel(img.shape, oversample=oversample, fftshift=False)
    return fftconv(img, kernel, normalize_kernel=False, fft_array=True,
                   fft_kernel=False)


def gauss(x1, x2, sigma, oversample=1, indexing='ij', normalize=False):
    """2D Gaussian function

    Parameters
    ----------
    x1, x2 : array_like
        1-D arrays representing the grid coordinates
    sigma : float or (2,) array_like
        Standard deviation of Gaussian. Providing two values allows for
        non-symmetric Gaussian interpreted as `(sigma_row, sigma_col)`
    oversample : float, optional
        Oversampling factor. Defailt is 1.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of output.
    normalize : bool, optional
        If True, the output is normalized such that its sum is equal to 1.
        If False (default), the output has max equal to one.

    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> r = np.arange(-128,128)
        >>> c = np.arange(-128,128)
        >>> g = prtools.gauss(r, c, sigma=20)
        >>> plt.imshow(g, cmap='gray')
    """
    xx1, xx2 = np.meshgrid(x1, x2, indexing=indexing)
    sigma = np.broadcast_to(sigma, (2,)) / oversample
    g = np.exp(-((xx1**2/(2*sigma[0]**2)) + (xx2**2/(2*sigma[1]**2))))
    if normalize:
        g = g / (2*np.pi * np.prod(sigma))
    return g


def sinc(x1, x2, indexing='ij'):
    r"""Normalized 2D sinc function

    Parameters
    ----------
    x1, x2 : array_like
        1-D arrays representing the grid coordinates
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of output.
    normalize : bool, optional
        If True, the output is normalized such that its sum is equal to 1.
        If False (default), the output has max equal to one.

    Returns
    -------
    ndarray

    See also
    --------
    :func:`~prtools.pixel_kernel`

    Notes
    -----
    This function uses ``numpy.sinc``, which computes the sinc function as

    .. math::
        \mbox{sinc}(x) = \frac{\sin \pi x}{\pi x}

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> r = np.linspace(-3,3,512)
        >>> c = np.linspace(-3,3,512)
        >>> s = prtools.sinc(r, c)
        >>> plt.imshow(s, cmap='gray', extent=[r.min(), r.max(), c.min(), c.max()])
    """
    xx1, xx2 = np.meshgrid(x1, x2, indexing=indexing)
    return np.sinc(xx1) * np.sinc(xx2)


def gauss_kernel(shape, sigma, oversample=1, fftshift=True):
    """2D Gaussian filter kernel

    This function returns the transfer function of a normalized 2D Gaussian
    function.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the kernel
    sigma : float
        Standard deviation of Gaussian. Providing two values allows for
        non-symmetric Gaussian interpreted as `(sigma_row, sigma_col)`
    oversample : float, optional
        Oversampling factor to represent in the kernel. Default is 1.
    fftshift : bool, optional
        If True (default), the kernel is FFT-shifted before it is returned.

    Returns
    -------
    ndarray

    See also
    --------
    :func:`~prtools.fftconv`
    :func:`~prtools.gauss_blur`
    :func:`~prtools.gauss`

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> blur_kernel = prtools.gauss_kernel((256,256), 1)
        >>> plt.imshow(blur_kernel, cmap='gray')

    Note this is functionally equivalent to

    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> r = np.fft.fftfreq(256)
        >>> c = np.fft.fftfreq(256)
        >>> sigma = 1
        >>> blur_kernel = np.fft.fftshift(prtools.gauss(r, c, 1/(2*np.pi*sigma)))
        >>> plt.imshow(blur_kernel, cmap='gray')
    """
    
    shape = np.broadcast_to(shape, (2,))
    sigma = np.broadcast_to(sigma, (2,))

    x1 = np.fft.fftfreq(shape[0])
    x2 = np.fft.fftfreq(shape[1])

    k = gauss(x1, x2, 1/(2*np.pi*sigma), oversample, indexing='ij',
              normalize=False)

    if fftshift:
        return np.fft.fftshift(k)
    else:
        return k


def pixel_kernel(shape, oversample=1, fftshift=True):
    r"""2D pixel MTF filter kernel
    
    This function returns a normalized 2D sinc function sized to represent
    the transfer function of an idealized square pixel.

    Parameters
    ----------
    shape : int or tuple of int
        Shape of the kernel
    oversample : float, optional
        Oversampling factor to represent in the kernel. Default is 1.
    fftshift : bool, optional
        If True (default), the kernel is FFT-shifted before it is returned.

    Returns
    -------
    ndarray

    See also
    --------
    :func:`~prtools.fftconv`
    :func:`~prtools.pixelate`
    :func:`~prtools.sinc`

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> pixel_mtf = prtools.pixel_kernel((256,256), 1)
        >>> plt.imshow(pixel_mtf, cmap='gray', vmin=0)

    Note this is functionally equivalent to

    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> r = np.fft.fftfreq(256)
        >>> c = np.fft.fftfreq(256)
        >>> pixel_mtf = np.fft.fftshift(np.abs(prtools.sinc(r, c)))
        >>> plt.imshow(pixel_mtf, cmap='gray', vmin=0)
    """
    shape = np.broadcast_to(shape, (2,))

    x1 = np.fft.fftfreq(shape[0]) / oversample
    x2 = np.fft.fftfreq(shape[1]) / oversample

    k = np.abs(sinc(x1, x2, indexing='ij'))

    if fftshift:
        return np.fft.fftshift(k)
    else:
        return k
