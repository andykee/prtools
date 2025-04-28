from prtools import __backend__
from prtools._backend import numpy as np


def mesh(shape, shift=(0, 0), angle=0, indexing='ij'):
    """Generate a standard mesh.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    shift : (2,) array_like, optional
        Coordinate translation from center of containing array.
        Default is (0, 0)
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of mesh.

    Returns
    -------
    list of ndarrays

    """
    shape = np.broadcast_to(np.asarray(shape), (2,))
    angle = np.radians(angle)
    if indexing == 'xy':
        x1, x2 = _meshxy(shape, shift, angle)
    elif indexing == 'ij':
        x1, x2 = _meshij(shape, shift, angle)
    else:
        raise ValueError(
            "Valid values for indexing are 'xy' and 'ij'.")
    return x1, x2


def _meshxy(shape, shift, angle):
    xx, yy = np.meshgrid((np.arange(shape[1])-np.floor(shape[1]/2))-shift[0],
                         (np.floor(shape[0]/2)-np.arange(shape[0])-shift[1]))

    x = xx * np.cos(angle) + yy * np.sin(angle)
    y = xx * -np.sin(angle) + yy * np.cos(angle)

    return x, y


def _meshij(shape, shift, angle):
    nr = shape[0]
    nc = shape[1]
    rr, cc = np.meshgrid(np.arange(nr) - np.floor(nr/2.0) - shift[0],
                         np.arange(nc) - np.floor(nc/2.0) - shift[1],
                         indexing='ij')
    r = rr * np.cos(angle) + cc * np.sin(angle)
    c = rr * -np.sin(angle) + cc * np.cos(angle)
    return r, c


def circle(shape, radius, shift=(0, 0), antialias=True, indexing='ij'):
    """Draw a circle

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    radius : float
        Radius of circle in pixels
    shift : (2,) array_like, optional
        How far to shift center in float (rows, cols). Default is (0, 0).
    antialias : bool, optional
        If True (default), the shape edges are antialiased.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of output.

    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> circ = prtools.circle(shape=(256, 256), radius=100)
        >>> plt.imshow(circ, cmap='gray')

    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> circ = prtools.circle(shape=(256, 256), radius=25, shift=(-80, 50))
        >>> plt.imshow(circ, cmap='gray')

    """
    rr, cc = mesh(shape, indexing=indexing)
    r = np.sqrt(np.square(rr - shift[0]) + np.square(cc - shift[1]))
    mask = np.clip(radius + 0.5 - r, 0.0, 1.0)
    if not antialias:
        if __backend__ == 'jax':
            mask = mask.at[mask > 0].set(1)
        else:
            mask[mask > 0] = 1
    return mask


def hexagon(shape, radius, shift=(0, 0), rotate=False, antialias=True):
    """Draw a hexagon

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    radius : float
        Radius of outscribing circle (which also equals the side length) in
        pixels.
    shift : tuple of floats, optional
        How far to shift center in (rows, cols). Default is (0, 0).
    rotate : bool, optional
        Rotate mask so that flat sides are aligned with the Y direction instead
        of the default orientation which is aligned with the X direction.
    antialias : bool, optional
        If True (default), the shape edges are antialiased.

    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> hex = prtools.hexagon(shape=(256,256), radius=75)
        >>> plt.imshow(hex, cmap='gray')

    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> hex = prtools.hexagon(shape=(256,256), radius=75, rotate=True)
        >>> plt.imshow(hex, cmap='gray')

    """
    r, c = mesh(shape, shift)
    mask = np.ones(shape)

    inner_radius = radius * np.sqrt(3)/2

    for n in range(6):

        theta = n * np.pi/3 if rotate else n * np.pi/3 + np.pi/6
        rho = r * np.sin(theta) + c * np.cos(theta)

        if antialias:
            slc = np.clip(inner_radius + 0.5 - rho, 0.0, 1.0)
        else:
            slc = np.ones(shape)
            slc[rho > inner_radius] = 0

        mask = np.minimum(mask, slc)

    return mask


def rectangle(shape, width, height, shift=(0, 0), angle=0, antialias=True,
              indexing='ij'):
    """Draw a rectangle

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    width : float
        Width of rectangle in pixels
    height : float
        Height of rectangle in pixels
    shift : tuple of floats, optional
        How far to shift center in (rows, cols). Default is (0, 0).
    angle : float, optional
        Rotation of rectangle in degrees counterclockwise from horizontal.
        Default is 0.
    antialias : bool, optional
        If True (default), the shape edges are antialiased.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of output.

    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> rect = prtools.rectangle(shape=(256,256), width=200, height=100)
        >>> plt.imshow(rect, cmap='gray')

    """
    rr, cc = mesh(shape, shift, angle)
    rect = np.ones(shape)

    width_clip = np.clip(0.5 + (width/2) - np.abs(cc), 0, 1)
    height_clip = np.clip(0.5 + (height/2) - np.abs(rr), 0, 1)

    rect = np.minimum(np.minimum(rect, width_clip), height_clip)

    if not antialias:
        rect[rect > 0] = 1

    return rect


def spider(shape, width, angle=0, shift=(0, 0), antialias=True, indexing='ij'):
    """Draw a spider

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    width : float
        Width of rectangle in pixels
    angle : float, optional
        Rotation of spider in degrees counterclockwise from horizontal.
        Default is 0.
    shift : tuple of floats, optional
        How far to shift center in (rows, cols). Default is (0, 0).
    antialias : bool, optional
        If True (default), the spider edges are antialiased.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of output.

    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> spider = prtools.spider(shape=(256,256), width=3, angle=30)
        >>> plt.imshow(spider, cmap='gray')

    """
    len = np.sqrt(2) * np.max(shape)/2  # max length when angle is a multiple of 45 deg
    shift_dist = len / 2
    shift_row = -shift_dist * np.sin(np.deg2rad(angle))
    shift_col = shift_dist * np.cos(np.deg2rad(angle))
    shift = (shift[0] + shift_row, shift[1] + shift_col)
    return 1 - rectangle(shape, len, width, shift, angle, antialias, indexing)


def ellipse():
    pass


def gauss(shape, sigma, shift=(0, 0), indexing='ij'):
    """Generate a 2D Gaussian function.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    sigma : float or (2,) array_like
        Stardard deviation of the Gaussian in pixels. If sigma has two
        entries it is interpreted as (sigma horizontal, sigma vertical).
    shift : (2,) array_like, optional
        Shape translation from center of containing array. Default is (0, 0)
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of output.

    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> gauss = prtools.gauss(shape=(256,256), sigma=20)
        >>> plt.imshow(gauss, cmap='gray')

    """
    sigma = np.broadcast_to(np.asarray(sigma), (2,))
    x, y = mesh(shape, shift=shift, angle=0, indexing=indexing)
    G = np.exp(-((x**2/(2*sigma[0]**2)) + (y**2/(2*sigma[1]**2))))
    G /= 2*np.pi * np.prod(sigma)  # normalization
    return G


def sin(shape, cycles, shift=(0, 0), angle=0, indexing='ij'):
    """Generate a 2D sine function.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    cycles : float
        Number of cycles represented across the shape.
    shift : (2,) array_like, optional
        Shape translation from center of containing array. Default is (0, 0)
    angle : float, optional
        Rotation in degrees from horizontal. Default is 0.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of output.

    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> sin = prtools.sin(shape=(256,256), cycles=5)
        >>> plt.imshow(sin, cmap='gray')

    """

    x = np.linspace(-cycles*np.pi, cycles*np.pi, shape[1])
    y = np.linspace(-cycles*np.pi, cycles*np.pi, shape[0])

    dx = cycles*2*np.pi/(shape[1]-1)
    dy = cycles*2*np.pi/(shape[0]-1)

    if indexing == 'xy':
        x -= shift[1] * dx
        y += shift[0] * dy
    elif indexing == 'ij':
        x -= shift[0] * dy
        y -= shift[1] * dx
    else:
        raise ValueError(
            "Valid values for indexing are 'xy' and 'ij'.")

    X, Y = np.meshgrid(x, y)
    Z = X*np.cos(-np.radians(angle)) + Y*np.sin(-np.radians(angle))
    return np.sin(Z)


def waffle(shape, cycles, shift=(0, 0), angle=0, indexing='ij'):
    """
    Generate a 2D waffle function.

    The waffle function is the sum of two orthogonal sine functions.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    cycles : float
        Number of cycles represented across the shape.
    shift : (2,) array_like, optional
        Shape translation from center of containing array. Default is (0, 0)
    angle : float, optional
        Rotation in degrees from horizontal. Default is 0.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of output.

    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> waffle = prtools.waffle(shape=(256,256), cycles=10)
        >>> plt.imshow(waffle, cmap='gray')


    """

    a = sin(shape, cycles, shift, angle+45, indexing)
    b = sin(shape, cycles, shift, angle+135, indexing)
    return a + b
