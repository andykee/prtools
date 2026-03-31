from array_api_compat import is_jax_namespace as is_jax
import numpy as np

from prtools._array_api import array_namespace, xp_multi_dot_three


def dft2(f, alpha, shape=None, shift=(0, 0), offset=(0, 0), axes=(-2, -1),
         unitary=True, out=None):
    r"""Compute the 2-dimensional discrete Fourier Transform.

    The DFT is defined in one dimension as

    .. math::

        A_k = \sum_{m=0}^{n-1} a_n \exp \left(-2\pi i\frac{mk}{n}\right) \hspace{3em} k=0, \dots, n-1

    This function allows independent control over input shape, output shape,
    and output sampling by implementing the matrix triple product algorithm
    described in [1].

    Parameters
    ----------
    f : array_like
        2D array to Fourier Transform
    alpha : float or array_like
        Output plane sampling interval (frequency). If :attr:`alpha` is an
        array, ``alpha[1]`` represents row-wise sampling and ``alpha[2]``
        represents column-wise sampling. If :attr:`alpha` is a scalar,
        ``alpha[1] = alpha[2] = alpha`` gives uniform sampling across the rows
        and columns of the output plane.
    shape : (2,) array_like, optional
        Size of the output array :attr:`F`. If :attr:`shape` is an array,
        ``F.shape = (shape[0], shape[1])``. Default is ``f.shape``.
    shift : array_like, optional
        Number of pixels in (r,c) to shift the DC pixel in the output plane
        with the origin centrally located in the plane. Default is ``(0, 0)``.
    offset : array_like, optional
        Number of pixels in (r,c) that the input plane is shifted relative to
        the origin. Default is ``(0, 0)``.
    axes : (2,) array_like of ints, optional
        Axes over which to compute the DFT. If not given, the last two axes are
        used.
    unitary : bool, optional
        Normalization flag. If ``True``, a normalization is performed on the
        output such that the DFT operation is unitary and energy is conserved
        through the Fourier transform operation (Parseval's theorem). In this
        way, the energy in in a limited-area DFT is a fraction of the total
        energy corresponding to the limited area. Default is ``True``.
    out : ndarray or None
        A location into which the result is stored. If provided, ``out.shape ==
        shape`` and ``out.dtype == np.complex``. If not provided or None, a
        freshly-allocated array is returned.

    Returns
    -------
    F : complex ndarray

    Notes
    -----
    * Setting ``alpha = 1/f.shape`` and ``shape = f.shape`` is equivalent to

      ::

          F = np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(f)))

    * ``dft2()`` is designed to place the DC pixel in the same location as a
      well formed call to any standard FFT for both even and odd sized input
      arrays. The DC pixel is located at ``np.floor(shape/2) + 1``, which is
      consistent with calls to Numpy's FFT method where the input and output
      are correctly shifted:
      ``np.fft.ifftshift(np.fft.fft2(np.fft.fftshift(f)))``.

    * If the y-axis shift behavior is not what you are expecting, you most
      likely have your plotting axes flipped (matplotlib's default behavior is
      to place [0,0] in the upper left corner of the axes). This may be
      resolved by either flipping the sign of the y component of ``shift`` or
      by passing ``origin = 'lower'`` to ``imshow()``.

    References
    ----------
    [1] Soummer, et. al. Fast computation of Lyot-style coronagraph
    propagation (2007)
    """

    return _dftcore(f, alpha, shape, shift, offset, axes, unitary,
                    forward=True, out=out)


def _dftcore(f, alpha, shape, shift, offset, axes, unitary, forward, out):

    xp = array_namespace(f)

    if out is not None:
        if is_jax(xp):
            raise ValueError('JAX backend does not support the out parameter')
        else:
            if not xp.can_cast(complex, out.dtype):
                raise TypeError(f"Cannot cast complex output to dtype('{out.dtype}')")

    f = xp.asarray(f)

    (M, N), axes = _cook_nd_args(f, shape, axes)
    m, n = (f.shape[axes[0]], f.shape[axes[1]])

    # note we have to explicitly cast these objects to xp.array objects to
    # work around a jax issue (bug?)
    alpha_row, alpha_col = xp.broadcast_to(xp.asarray(alpha), (2,))
    shift_row, shift_col = xp.broadcast_to(xp.asarray(shift), (2,))
    offset_row, offset_col = xp.broadcast_to(xp.asarray(offset), (2,))

    E1, E2 = _dft2_matrices(m, n, M, N, alpha_row, alpha_col, shift_row,
                            shift_col, offset_row, offset_col, forward, xp=xp)
    F = xp_multi_dot_three(E1, f, E2, axes, out, xp=xp)

    if unitary:
        out = None if is_jax(xp) else F
        F = xp.multiply(F, xp.sqrt(xp.abs(alpha_row * alpha_col)), out=out)

    # idft rescaling
    if not forward:
        out = None if is_jax(xp) else F
        F = xp.divide(F, f.size, out=out)

    return F


def _dft2_matrices(m, n, M, N, alphar, alphac, shiftr, shiftc, offsetr,
                   offsetc, forward, xp):
    if forward:
        c = -1j
    else:
        c = 1j
    R, S, U, V = _dft2_coords(m, n, M, N, xp)
    E1 = xp.exp(2.0 * c * xp.pi * alphar * xp.outer(R+offsetr, U-shiftr)).T
    E2 = xp.exp(2.0 * c * xp.pi * alphac * xp.outer(S+offsetc, V-shiftc))
    return E1, E2


def _dft2_coords(m, n, M, N, xp):
    # R and S are (r,c) coordinates in the (m x n) input plane f
    # V and U are (r,c) coordinates in the (M x N) output plane F
    R = xp.arange(m) - xp.floor(m/2.0)
    S = xp.arange(n) - xp.floor(n/2.0)
    U = xp.arange(M) - xp.floor(M/2.0)
    V = xp.arange(N) - xp.floor(N/2.0)

    return R, S, U, V


def _cook_nd_args(a, s=None, axes=None):
    # slightly modified version of numpy's function of the same name
    if s is None:
        if axes is None:
            if a.ndim == 2:
                s = list(a.shape)
            elif a.ndim == 3:
                s = list(a.shape[1:3])
            else:
                raise ValueError("Array must have ndim == 2 or 3")
        else:
            s = np.take(np.asarray(a.shape), np.asarray(axes))
    s = list(s)
    if axes is None:
        axes = list(range(-len(s), 0))
    if len(s) != len(axes):
        raise ValueError("Shape and axes have different lengths.")
    return s, axes


def idft2(F, alpha, shape=None, shift=(0, 0), offset=(0, 0), axes=(-2, -1),
          unitary=True, out=None):
    r"""Compute the 2-dimensional inverse discrete Fourier Transform.

    The IDFT is defined in one dimension as

    .. math::

        a_m = \frac{1}{n}\sum_{k=0}^{n-1} A_k \exp \left(2\pi i\frac{mk}{n}\right) \hspace{3em} m=0, \dots, n-1

    This function allows independent control over input shape, output shape,
    and output sampling by implementing the matrix triple product algorithm
    described in [1].

    Parameters
    ----------
    F : array_like
        2D array to Fourier Transform
    alpha : float or array_like
        Input plane sampling interval (frequency). If :attr:`alpha` is an
        array, ``alpha[1]`` represents row-wise sampling and ``alpha[2]``
        represents column-wise sampling. If :attr:`alpha` is a scalar,
        ``alpha[1] = alpha[2] = alpha`` represents uniform sampling across the
        rows and columns of the input plane.
    shape : int or array_like, optional
        Size of the output array :attr:`F`. If :attr:`npshapeix` is an array,
        ``F.shape = (shape[0], shape[1])``. If :attr:`shape` is a scalar,
        ``F.shape = (shape, shape)``. Default is ``F.shape``
    shift : array_like, optional
        Number of pixels in (x,y) to shift the DC pixel in the output plane
        with the origin centrally located in the plane. Default is `[0,0]`.
    offset : array_like, optional
        Number of pixels in (r,c) that the input plane is shifted relative to
        the origin. Default is ``(0, 0)``.
    axes : (2,) array_like of ints, optional
        Axes over which to compute the DFT. If not given, the last two axes are
        used.
    unitary : bool, optional
        Normalization flag. If ``True``, a normalization is performed on the
        output such that the DFT operation is unitary and energy is conserved
        through the Fourier transform operation (Parseval's theorem). In this
        way, the energy in in a limited-area DFT is a fraction of the total
        energy corresponding to the limited area. Default is ``True``.
    out : ndarray or None
        A location into which the result is stored. If provided, ``out.shape ==
        shape`` and ``out.dtype == np.complex``. If not provided or None, a
        freshly-allocated array is returned.

    Returns
    -------
    f : complex ndarray

    Notes
    -----
    * Setting ``alpha = 1/F.shape`` and ``shape = F.shape`` is equivalent to

      ::

          F = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(F)))

    * ``idft2()`` is designed to place the DC pixel in the same location as a
      well formed call to any standard FFT for both even and odd sized input
      arrays. The DC pixel is located at ``np.floor(shape/2) + 1``, which is
      consistent with calls to Numpy's IFFT method where the input and output
      are correctly shifted:
      ``np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(f)))``.

    * If the y-axis shift behavior is not what you are expecting, you most
      likely have your plotting axes flipped (matplotlib's default behavior is
      to place [0,0] in the upper left corner of the axes). This may be
      resolved by either flipping the sign of the y component of ``shift`` or
      by passing ``origin = 'lower'`` to ``imshow()``.

    References
    ----------
    [1] Soummer, et. al. Fast computation of Lyot-style coronagraph
    propagation (2007)

    """
    return _dftcore(F, alpha, shape, shift, offset, axes, unitary,
                    forward=False, out=out)
