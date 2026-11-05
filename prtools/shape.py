import numpy as np

from prtools.mesh import mesh


def circle(shape, radius, shift=(0, 0), antialias=True, indexing='ij'):
    """Draw a circle

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    radius : float
        Radius of circle in pixels
    shift : (2,) array_like, optional
        How far to shift center. Default is (0, 0).
    antialias : bool, optional
        If True (default), the shape edges are antialiased.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of ``shift``.

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
    x1, x2 = mesh(shape, shift=shift, indexing=indexing)
    r = np.sqrt(np.square(x1) + np.square(x2))
    mask = np.clip(radius + 0.5 - r, 0.0, 1.0)
    if not antialias:
        mask[mask > 0] = 1
        #if __backend__ == 'jax':
        #    mask = mask.at[mask > 0].set(1)
        #else:
        #    mask[mask > 0] = 1
    return mask


def hexagon(shape, radius, shift=(0, 0), rotate=True, antialias=True,
            indexing='ij'):
    """Draw a hexagon

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    radius : float
        Radius of outscribing circle (which also equals the side length) in
        pixels.
    shift : tuple of floats, optional
        How far to shift center. Default is (0, 0).
    rotate : bool, optional
        If True (default), the mask is rotated so that the flat sides are
        aligned with the horizontal direction.
    antialias : bool, optional
        If True (default), the shape edges are antialiased.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of ``shift``.

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

        >>> hex = prtools.hexagon(shape=(256,256), radius=75, rotate=False)
        >>> plt.imshow(hex, cmap='gray')

    """
    shape = np.broadcast_to(shape, (2,))
    x1, x2 = mesh(shape, shift=shift, indexing=indexing)

    if indexing == 'ij':
        x1, x2 = x2, x1

    mask = np.ones(shape)

    inner_radius = radius * np.sqrt(3)/2

    for n in range(6):

        theta = n * np.pi/3 if rotate else n * np.pi/3 + np.pi/6
        rho = x1 * np.sin(theta) + x2 * np.cos(theta)

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
        How far to shift center. Default is (0, 0).
    angle : float, optional
        Mesh rotation in degrees. When ``indexing = 'xy'`` the rotation is
        measured from the +x axis toward the +y axis. When ``indexing = 'ij'``
        the rotation is measured from the +i axis toward the +j axis. Note
        that rotations occur about the shifted center if ``shift`` is nonzero.
        Default is 0.
    antialias : bool, optional
        If True (default), the shape edges are antialiased.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of ``shift``.

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
    x1, x2 = mesh(shape, shift=shift, angle=angle, indexing=indexing)

    if indexing == 'xy':
        # mesh returns (x, y) = (col, row) for 'xy' indexing
        x1, x2 = x2, x1

    rect = np.ones(shape)

    width_clip = np.clip(0.5 + (width/2) - np.abs(x2), 0, 1)
    height_clip = np.clip(0.5 + (height/2) - np.abs(x1), 0, 1)

    rect = np.minimum(np.minimum(rect, width_clip), height_clip)

    if not antialias:
        rect[rect > 0] = 1

    return rect


def ellipse(shape, xradius, yradius, shift=(0, 0), angle=0, antialias=True,
            indexing='ij'):
    """Draw an ellipse

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    xradius : float
        Semi-axis of the ellipse in pixels in the x (horizontal) direction.
    yradius : float
        Semi-axis of the ellipse in pixels in the y (vertical) direction.
    shift : (2,) array_like, optional
        How far to shift center. Default is (0, 0).
    angle : float, optional
        Mesh rotation in degrees. When ``indexing = 'xy'`` the rotation is
        measured from the +x axis toward the +y axis. When ``indexing = 'ij'``
        the rotation is measured from the +i axis toward the +j axis. Note
        that rotations occur about the shifted center if ``shift`` is nonzero.
        Default is 0.
    antialias : bool, optional
        If True (default), the shape edges are antialiased.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of ``shift``.

    Returns
    -------
    ndarray

    Examples
    --------
    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> ell = prtools.ellipse(shape=(256, 256), xradius=100, yradius=50)
        >>> plt.imshow(ell, cmap='gray')

    .. plot::
        :include-source:
        :context: reset
        :scale: 50

        >>> ell = prtools.ellipse(shape=(256, 256), xradius=100, yradius=50, angle=30)
        >>> plt.imshow(ell, cmap='gray')

    """
    if indexing == 'ij':
        a, b = yradius, xradius
    elif indexing == 'xy':
        a, b = xradius, yradius
    x1, x2 = mesh(shape, shift=shift, angle=angle, indexing=indexing)

    rho =np.sqrt((x1 / a)**2 + (x2 / b)**2)
    grad = np.sqrt((x1 / a**2)**2 + (x2 / b**2)**2)

    with np.errstate(divide='ignore', invalid='ignore'):
        dist = (1 - rho) * rho / grad
    dist = np.where(grad == 0, 1.0, dist)  # center is always interior
    ell = np.clip(dist + 0.5, 0.0, 1.0)

    if not antialias:
        ell[ell > 0] = 1

    return ell


def spider(shape, width, angle=0, shift=(0, 0), antialias=True, indexing='ij'):
    """Draw a spider

    Parameters
    ----------
    shape : array_like
        Size of output in pixels (nrows, ncols)
    width : float
        Width of rectangle in pixels
    angle : float, optional
        Mesh rotation in degrees. When ``indexing = 'xy'`` the rotation is
        measured from the +x axis toward the +y axis. When ``indexing = 'ij'``
        the rotation is measured from the +i axis toward the +j axis. Note
        that rotations occur about the shifted center if ``shift`` is nonzero.
        Default is 0.
    shift : tuple of floats, optional
        How far to shift center. Default is (0, 0).
    antialias : bool, optional
        If True (default), the spider edges are antialiased.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of ``shift``.

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
    shape = np.broadcast_to(shape, (2,))
    len = 2*np.max(shape)

    if indexing == 'xy':
        shift = [shift[1], shift[0]]
        angle = -angle

    shift_dist = len / 2
    shift_row = -shift_dist * np.sin(np.deg2rad(angle))
    shift_col = shift_dist * np.cos(np.deg2rad(angle))

    shift = (shift[0] + shift_row, shift[1] + shift_col)
    return 1 - rectangle(shape, width=len, height=width, shift=shift,
                         angle=angle, antialias=antialias,
                         indexing='ij')


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
        Matrix ('ij', default) or cartesian ('xy') indexing of ``shift``.

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
    r, c = mesh(shape, shift=shift, indexing=indexing, angle=0)
    G = np.exp(-((c**2/(2*sigma[0]**2)) + (r**2/(2*sigma[1]**2))))
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
        Mesh rotation in degrees. When ``indexing = 'xy'`` the rotation is
        measured from the +x axis toward the +y axis. When ``indexing = 'ij'``
        the rotation is measured from the +i axis toward the +j axis. Note
        that rotations occur about the shifted center if ``shift`` is nonzero.
        Default is 0.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of ``shift``.

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
        shift_r, shift_c = shift[1], shift[0]
        angle = -angle
    elif indexing == 'ij':
        shift_r, shift_c = shift[0], shift[1]
    else:
        raise ValueError(
            "Valid values for indexing are 'xy' and 'ij'.")

    x -= shift_c * dx
    y -= shift_r * dy

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
        Mesh rotation in degrees. When ``indexing = 'xy'`` the rotation is
        measured from the +x axis toward the +y axis. When ``indexing = 'ij'``
        the rotation is measured from the +i axis toward the +j axis. Note
        that rotations occur about the shifted center if ``shift`` is nonzero.
        Default is 0.
    indexing : {'ij', 'xy'}, optional
        Matrix ('ij', default) or cartesian ('xy') indexing of ``shift``.

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
    if indexing == 'xy':
        shift = np.flip(shift)
        angle = - angle
    a = sin(shape, cycles, shift, angle+45, indexing='ij')
    b = sin(shape, cycles, shift, angle+135, indexing='ij')
    return a + b
