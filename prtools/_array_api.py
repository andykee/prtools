import enum
import warnings

import array_api_compat
from array_api_compat import is_jax_namespace as is_jax
import numpy as np


def use(backend):
    msg = 'prtools.use() was deprecated in prtools v2.0.0. The numerical ' \
          'backend is now selected automatically depending on input array ' \
          'type or by specifying the xp argument (when available). '
    warnings.warn(msg + 'In a future prtools release this will be an error.',
                  category=DeprecationWarning, stacklevel=2)


class _ArrayCoerce(enum.Enum):
    none = 0
    numpy = 1


def _array_cls(array):
    if issubclass(type(array), list | tuple):
        return _ArrayCoerce.numpy
    else:
        return _ArrayCoerce.none


def array_namespace(*xs):
    """Get the array API compatible namespace for the arrays `xs`

    Notes
    -----
    This function is a wrapper arpund `array_api_compat.array_namespace`. It
    is inspired by the scipy function of the same name.
    """
    arrays = [np.array(x) if _array_cls(x) is _ArrayCoerce.numpy else x for x in xs]
    return array_api_compat.array_namespace(*arrays)


def scipy_namespace(xp):
    """Return the `scipy`-like namespace corresponding to the array namespace
    ``xp``.
    """
    if is_jax(xp):
        import jax
        return jax.scipy
    else:
        import scipy
        return scipy


def xp_multi_dot_three(a, b, c, axes, out, xp=None):
    if xp is None:
        xp = array_namespace(a, b, c)

    if is_jax(xp):
        # a few notes:
        # * while numpy-based implementation of this method is based on
        #   np.matmul, jax.numpy.matmul doesn't implement the axes argument so
        #   we have to use jax.numpy.linalg.multi_dot instead
        # * the implementation used here supports b with ndim in (2, 3)
        #   iterating over any of the 3 axes when b.ndim == 3
        # * jax.vmap handles the case when b.ndim == 3 compared with the numpy
        #   equivalent of this function which does everything within the
        #   confines of matmul using the axes argument
        if b.ndim == 2:
            return xp.linalg.multi_dot((a, b, c))
        else:
            import jax
            iter_axis = _iter_axis(axes)
            return jax.vmap(_multi_dot, in_axes=(None, iter_axis, None), out_axes=iter_axis)(a, b, c)
    else:
        # a few notes:
        # * this method is similar to np.linalg.multi_dot although it is less
        #   general - here we only consider the matrix triple product used as
        #   a part of prtools.dft2
        # * because we use np.matmul instead of np.linalg.multi_dot, we can
        #   take advantage of broadcasting a and c when b.ndim = 3. This
        #   eliminates a for loop in the code
        # * the implementation used here supports b with ndim in (2, 3)
        #   iterating over any of the 3 axes when b.ndim == 3
        # * the implementation used here is actually slightly faster than
        #   an equivalent call to np.linalg.multi_dot when b.ndim == 2
        # * np.linalg.multi_dot chooses the fastest multiplication order from
        #   [(ab)c, a(bc)] depending on the shapes of a, b, and c. There is no
        #   difference when computing the dft because both a and c are square
        #   matrices
        ab = xp.matmul(a, b, axes=[(0, 1), axes, axes])
        out = xp.matmul(ab, c, axes=[axes, (0, 1), axes], out=out)
        return out
    

def _multi_dot(a, b, c):
    # wrapper function to support vmap call signature
    xp = array_namespace(a, b, c)
    return xp.linalg.multi_dot((a, b, c))


def _iter_axis(axes):
    # pure Python to avoid dealing with JAX array mutability issues
    mask = [0, 1, 2]
    for ax in axes:
        mask[ax] = None
    return [ax for ax in mask if ax is not None][0]
