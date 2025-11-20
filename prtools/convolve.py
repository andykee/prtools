from prtools.backend import numpy as np

# gauss_kernel
# pixel_kernel
# fftconv(array, kernel)
# pixelate
# gauss_blur


def fftconv(array, kernel, normalize_kernel=True, fft_array=True,
            fft_kernel=True):

    a = np.fft.fft2(array) if fft_array else array
    k = np.fft.fft2(array) if fft_kernel else kernel

    if normalize_kernel:
        k /= np.sum(k)

    return np.fft.ifft2(a*k).real


def gauss_blur(img, sigma, oversample):
    pass

def gauss(x1, x2, sigma, oversample=1, indexing='ij', normalize=False):
    """2D Gaussian function

    Parameters
    ----------
    x1, x1 : array_like
        1-D arrays representing the grid coordinates
    sigma : float or (2,) array_like
        Standard deviation of Gaussian. Providing two values allows for
        non-symmetric Gaussian interpreted as `(sigma_row, sigma_col)`https://shokz.com/pages/openrunpro2
    oversample : int, optional
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

    """
    xx1, xx2 = np.meshgrid(x1, x2, indexing=indexing)
    sigma = np.broadcast_to(sigma, (2,)) / oversample
    g = np.exp(-((xx1**2/(2*sigma[0]**2)) + (xx2**2/(2*sigma[1]**2))))
    if normalize:
        g = g / (2*np.pi * np.prod(sigma))
    return g


def sinc(x1, x2, size=1, oversample=1, indexing='ij'):

    xx1, xx2 = np.meshgrid(x1, x2, indexing=indexing)
    size = np.broadcast_to(size, (2,)) / oversample
    return np.sinc(xx1 * size[0]) * np.sinc(xx2 * size[1])


def gauss_kernel(shape, sigma, oversample=1, fftshift=True):
    shape = np.broadcast_to(shape, (2,))
    sigma = np.broadcast_to(sigma, (2,))

    x1 = np.fft.fftfreq(shape[0]*oversample)
    x2 = np.fft.fftfreq(shape[1]*oversample)

    k = gauss(x1, x2, 1/(2*np.pi*sigma), oversample, indexing='ij',
              normalize=False)

    if fftshift:
        return np.fft.fftshift(k)
    else:
        return k


def pixel_kernel():
    pass
