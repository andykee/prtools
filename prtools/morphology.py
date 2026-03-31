from array_api_compat import is_jax_namespace as is_jax

from prtools._array_api import array_namespace, scipy_namespace


def binary_erosion(input, structure=None, iterations=1, mask=None,
                   output=None, border_value=0):
    """Binary erosion using the given structure element.
    
    Parameters
    ----------
    input : array_like
        Array to be eroded. Nonzero elements form the subset to be eroded.
    structure : array_like, optional
        Structure element used for erosion. Nonzero elements are considered 
        True. If None (default), a structure element with a square
        connectivity equal to one is used.
    iterations : int, optional
        Number of times to repeat the erosion. Default is 1.
    mask : array_like, optional
        Mask applied to the inputs where a True value indicates the 
        corresponding input element should be included in the erosion. 
    output : ndarray, optional
        Array location into which the result is stored. If None (default), a
        freshly-allocated array is returned. 

        .. note::

            ``output`` must be None if ``input`` is a JAX array.

    border_value : int (cast to 0 or 1), optional
        Value at the border of the output array.

    Returns
    -------
    binary_erosion : ndarray of bools
        Erosion of the input by the structure element.

    See Also
    --------
    :func:`binary_dilation`
    :func:`binary_opening`
    :func:`binary_closing`

    Notes
    -----
    This function is designed to be compatible with JAX ``jit`` and ``grad``
    operations.

    Examples
    --------
    .. code:: pycon

        >>> a = np.zeros((7,7), dtype=int)
        >>> a[1:6, 2:5] = 1
        >>> a
        array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]])

        >>> prtools.binary_erosion(a)
        array([[False, False, False, False, False, False, False],
               [False, False, False, False, False, False, False],
               [False, False, False,  True, False, False, False],
               [False, False, False,  True, False, False, False],
               [False, False, False,  True, False, False, False],
               [False, False, False, False, False, False, False],
               [False, False, False, False, False, False, False]])

        >>> prtools.binary_erosion(a).astype(int)
        array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 1, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]])

        >>> # Erosion removes objects smaller than the structure
        >>> prtools.binary_erosion(a, structure=np.ones((5,5))).astype(int)
        array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]])
    """
    xp = array_namespace(input)
    scipy = scipy_namespace(xp)

    if is_jax(xp):
        if output is not None:
            raise ValueError('jax does not support in-place operations')
        
        import jax

        if structure is None:
            structure = xp.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=input.dtype)
        else:
            structure = xp.asarray(structure, dtype=input.dtype)

        structure_sum = xp.sum(structure)

        def _erode_once(arr):
            pad_width = tuple((s // 2, s // 2) for s in structure.shape)
            padded = xp.pad(arr, pad_width, mode='constant',
                            constant_values=border_value)
            conv = scipy.signal.convolve(padded, structure,
                                             mode='valid')
            result = (conv >= structure_sum).astype(arr.dtype)
            if mask is not None:
                result = xp.where(mask, result, arr)
            return result

        result = jax.lax.fori_loop(0, iterations,
                                   lambda _, x: _erode_once(x), input)
        return result

    else:
        
        return scipy.ndimage.binary_erosion(input, structure=structure,
                                            iterations=iterations,
                                            mask=mask, output=output,
                                            border_value=border_value)


def binary_dilation(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0):
    """Binary dilation using the given structure element.
    
    Parameters
    ----------
    input : array_like
        Array to be dilated. Nonzero elements form the subset to be dilated.
    structure : array_like, optional
        Structure element used for dilation. Nonzero elements are considered
        True. If None (default), a structure element with a square
        connectivity equal to one is used.
    iterations : int, optional
        Number of times to repeat the dilation. Default is 1.
    mask : array_like, optional
        Mask applied to the inputs where a True value indicates the 
        corresponding input element should be included in the dilation. 
    output : ndarray, optional
        Array location into which the result is stored. If None (default), a
        freshly-allocated array is returned. 

        .. note::

            ``output`` must be None if ``input`` is a JAX array.

    border_value : int (cast to 0 or 1), optional
        Value at the border of the output array.

    Returns
    -------
    binary_dilation : ndarray of bools
        Dilation of the input by the structure element.

    See Also
    --------
    :func:`binary_erosion`
    :func:`binary_opening`
    :func:`binary_closing`

    Notes
    -----
    This function is designed to be compatible with JAX ``jit`` and ``grad``
    operations.

    Examples
    --------
    .. code:: pycon

        >>> a = np.zeros((5,5), dtype=int)
        >>> a[2,2] = 1
        >>> a
        array([[0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0]])
        
        >>> prtools.binary_dilation(a)
        array([[False, False, False, False, False],
               [False, False,  True, False, False],
               [False,  True,  True,  True, False],
               [False, False,  True, False, False],
               [False, False, False, False, False]])

        >>> prtools.binary_dilation(a).astype(int)
        array([[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]])

        >>> prtools.binary_dilation(a, iterations=2).astype(int)
        array([[0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [1, 1, 1, 1, 1],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0]])

        >>> prtools.binary_dilation(a, structure=np.ones((3,3))).astype(int)
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])
    """
    xp = array_namespace(input)
    scipy = scipy_namespace(xp)

    if is_jax(xp):

        if output is not None:
            raise ValueError('jax does not support in-place operations')

        import jax

        if structure is None:
            structure = xp.array([[0, 1, 0],
                                  [1, 1, 1],
                                  [0, 1, 0]], dtype=input.dtype)
        else:
            structure = xp.asarray(structure, dtype=input.dtype)

        def _dilate_once(arr):
            pad_width = tuple((s // 2, s // 2) for s in structure.shape)
            padded = xp.pad(arr, pad_width, mode='constant',
                            constant_values=border_value)
            conv = scipy.signal.convolve(padded, structure,
                                         mode='valid')
            result = (conv >= 1).astype(arr.dtype)
            if mask is not None:
                result = xp.where(mask, result, arr)
            return result

        result = jax.lax.fori_loop(0, iterations,
                                   lambda _, x: _dilate_once(x), input)
        return result

    else:

        return scipy.ndimage.binary_dilation(input, structure=structure,
                                             iterations=iterations,
                                             mask=mask, output=output,
                                             border_value=border_value)


def binary_closing(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0):
    """Binary closing using the given structure element.

    The closing of an input image is the erosion of the dilation of the image
    by the structure element.
    
    Parameters
    ----------
    input : array_like
        Array to be closed. Nonzero elements form the subset to be closed.
    structure : array_like, optional
        Structure element used for closing. Nonzero elements are considered
        True. If None (default), a structure element with a square
        connectivity equal to one is used.
    iterations : int, optional
        Number of times to repeat the closing. Default is 1.
    mask : array_like, optional
        Mask applied to the inputs where a True value indicates the 
        corresponding input element should be included in the closing. 
    output : ndarray, optional
        Array location into which the result is stored. If None (default), a
        freshly-allocated array is returned. 

        .. note::

            ``output`` must be None if ``input`` is a JAX array.

    border_value : int (cast to 0 or 1), optional
        Value at the border of the output array.

    Returns
    -------
    binary_closing : ndarray of bools
        Closing of the input by the structure element.

    See Also
    --------
    :func:`binary_dilation`
    :func:`binary_erosion`
    :func:`binary_opening`

    Notes
    -----
    This function is designed to be compatible with JAX ``jit`` and ``grad``
    operations.

    Examples
    --------
    .. code:: pycon

        >>> a = np.zeros((5,5), dtype=int)
        >>> a[1:-1, 1:-1] = 1; a[2,2] = 0
        >>> a
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 0, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])
        
        >>> # Closing removes small holes
        >>> prtools.binary_closing(a).astype(int)
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])

    .. code:: pycon

        >>> a = np.zeros((7,7), dtype=int)
        >>> a[1:6, 2:5] = 1; a[1:3,3] = 0
        >>> a
        array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 1, 0, 0],
               [0, 0, 1, 0, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]])

        >>> # Closing can also coarsen boundaries with fine hollows
        >>> prtools.binary_closing(a).astype(int)
        array([[0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 1, 1, 1, 0, 0],
               [0, 0, 0, 0, 0, 0, 0]])
    """
    tmp = binary_dilation(input, structure, iterations, mask,
                          output, border_value)
    return binary_erosion(tmp, structure, iterations, mask,
                          output, border_value)


def binary_opening(input, structure=None, iterations=1, mask=None,
                    output=None, border_value=0):
    """Binary opening using the given structure element.

    The opening of an input image is the dilation of the erosion of the image
    by the structure element.
    
    Parameters
    ----------
    input : array_like
        Array to be opened. Nonzero elements form the subset to be opened.
    structure : array_like, optional
        Structure element used for opening. Nonzero elements are considered
        True. If None (default), a structure element with a square
        connectivity equal to one is used.
    iterations : int, optional
        Number of times to repeat the opening. Default is 1.
    mask : array_like, optional
        Mask applied to the inputs where a True value indicates the 
        corresponding input element should be included in the opening. 
    output : ndarray, optional
        Array location into which the result is stored. If None (default), a
        freshly-allocated array is returned. 

        .. note::

            ``output`` must be None if ``input`` is a JAX array.

    border_value : int (cast to 0 or 1), optional
        Value at the border of the output array.

    Returns
    -------
    binary_opening : ndarray of bools
        Opening of the input by the structure element.

    See Also
    --------
    :func:`binary_dilation`
    :func:`binary_erosion`
    :func:`binary_closing`

    Notes
    -----
    This function is designed to be compatible with JAX ``jit`` and ``grad``
    operations.

    Examples
    --------
    .. code:: pycon

        >>> a = np.zeros((5,5), dtype=int)
        >>> a[1:4, 1:4] = 1; a[4, 4] = 1
        >>> a
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 1]])
        
        >>> # Opening removes small objects
        >>> prtools.binary_opening(a, structure=np.ones((3,3))).astype(int)
        array([[0, 0, 0, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 0, 0, 0]])

        >>> # Opening can also smooth corners
        >>> prtools.binary_opening(a).astype(int)
        array([[0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0],
               [0, 1, 1, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 0, 0, 0]])
    """
    tmp = binary_erosion(input, structure, iterations, mask,
                          output, border_value)
    return binary_dilation(tmp, structure, iterations, mask,
                          output, border_value)