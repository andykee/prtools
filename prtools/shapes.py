import numpy as np


def circle(shape, radius, shift=(0, 0), indexing='xy'):
    """Compute a circle with anti-aliasing.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    radius : float
        Radius of circle in pixels
    shift : (2,) array_like, optional
        Shape translation from center of containing array. Default is (0, 0)
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of shift.

    Returns
    -------
    out : ndarray

    See Also
    --------
    circlemask : Compute a circular mask.

    """
    x, y = _mesh(shape, shift=shift, rotate=0, indexing=indexing)
    r = np.sqrt(np.square(x) + np.square(y))
    return np.clip(radius + 0.5 - r, 0.0, 1.0)


def circlemask(shape, radius, shift=(0, 0), indexing='xy'):
    """Compute a circular mask.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    radius : float
        Radius of circle in pixels
    shift : (2,) array_like, optional
        Shape translation from center of containing array. Default is (0, 0)
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of shift.

    Returns
    -------
    out : ndarray

    See Also
    --------
    circle : Compute a circle with anti-aliasing.

    """
    x, y = _mesh(shape, shift=shift, rotate=0, indexing=indexing)
    r = np.sqrt(np.square(x) + np.square(y))
    mask = np.zeros(shape)
    mask[r < radius] = 1
    return mask


def hexagon(shape, radius, rotate=False):
    """Compute a hexagon mask.

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    radius : int
        Radius of outscribing circle (which also equals the side length) in
        pixels.
    rotate : bool
        Rotate mask so that flat sides are aligned with the Y direction instead
        of the default orientation which is aligned with the X direction.

    Returns
    -------
    out : ndarray

    """

    inner_radius = radius * np.sqrt(3)/2
    side_length = radius/2

    y, x = _mesh(shape)

    rect = np.where((np.abs(x) <= side_length) & (np.abs(y) <= inner_radius))
    left_tri = np.where((x <= -side_length) & (x >= -radius) & (np.abs(y) <= (x + radius)*np.sqrt(3)))
    right_tri = np.where((x >= side_length) & (x <= radius) & (np.abs(y) <= (radius - x)*np.sqrt(3)))

    mask = np.zeros(shape)
    mask[rect] = 1
    mask[left_tri] = 1
    mask[right_tri] = 1

    if rotate:
        return mask.transpose()
    else:
        return mask


def rectangle(shape, width, height, shift=(0,0), rotate=0, indexing='xy'):
    """Generate a rectangle with anti-ailiasing.
    
    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    width : float
        Horizontal extent of rectangle in pixels
    height : float
        Vertical extent of rectangle in pixels
    shift : (2,) array_like, optional
        Shape translation from center of containing array. Default is (0, 0)
    rotate : float, optional
        Rotation of rectangle in degrees from horizontal. Default is 0.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of shift.

    Returns
    -------
    rectangle : ndarray

    """
    x, y = _mesh(shape, shift=shift, rotate=rotate, indexing=indexing)
    xx = np.clip(0.5 + (width/2) - np.abs(x), 0, 1)
    yy = np.clip(0.5 + (height/2) - np.abs(y), 0, 1)

    rect = np.ones(shape)
    rect = np.minimum(np.minimum(rect, xx), yy)
    return rect


def ellipse():
    pass


def gauss2(shape, sigma, shift=(0,0), indexing='xy'):
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
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of shift.

    Returns
    -------
    gauss2 : ndarray
    
    """
    sigma = np.broadcast_to(np.asarray(sigma), (2,))
    x, y = _mesh(shape, shift=shift, rotate=0, indexing=indexing)
    G = np.exp(-((x**2/(2*sigma[0]**2)) + (y**2/(2*sigma[1]**2))))
    G /= 2*np.pi * np.prod(sigma)  # normalization
    return G


def sin2(shape, cycles, shift=(0,0), rotate=0, indexing='xy'):
    """Generate a 2D sine function.
    
    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    cycles : float
        Number of cycles represented across the shape.
    shift : (2,) array_like, optional
        Shape translation from center of containing array. Default is (0, 0)
    rotate : float, optional
        Rotation in degrees from horizontal. Default is 0.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of shift.

    Returns
    -------
    sin2 : ndarray

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
    Z = X*np.cos(-np.radians(rotate)) + Y*np.sin(-np.radians(rotate))
    return np.sin(Z)


def waffle2(shape, cycles, shift=(0,0), rotate=0, indexing='xy'):
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
    rotate : float, optional
        Rotation in degrees from horizontal. Default is 0.
    indexing : {'xy', 'ij'}, optional
        Cartesian ('xy', default) or matrix ('ij') indexing of shift.

    Returns
    -------
    waffle2 : ndarray
    """

    a = sin2(shape, cycles, shift, rotate+45, indexing)
    b = sin2(shape, cycles, shift, rotate+135, indexing)
    return a + b


def _mesh(shape, shift=(0, 0), rotate=0, indexing='xy'):
    """Generate a standard mesh."""

    angle = np.radians(rotate)        
    
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
    rr, cc = np.meshgrid(np.arange(shape[1]) - np.floor(shape[1]/2.0) - shift[1],
                         np.arange(shape[0]) - np.floor(shape[0]/2.0) - shift[0])
    
    r = rr * np.cos(-angle) + cc * np.sin(-angle)
    c = rr * -np.sin(-angle) + cc * np.cos(-angle)

    return r, c