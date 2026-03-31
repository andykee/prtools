from array_api_compat import is_jax_namespace as is_jax

from prtools._array_api import array_namespace, scipy_namespace


def binary_erosion(input, structure=None, iterations=1, mask=None,
                   output=None, border_value=0):
    xp = array_namespace(input)
    scipy = scipy_namespace(xp)

    if is_jax(xp):

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
    xp = array_namespace(input)
    scipy = scipy_namespace(xp)

    if is_jax(xp):

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
