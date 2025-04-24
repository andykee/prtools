import jax.numpy
import jax.scipy

from prtools import *
import prtools.backend

from prtools.jax import optimize

__all__ = prtools.__all__ + ['optimize']

prtools.backend.set_backend('jax',
                            numpy_module=jax.numpy,
                            scipy_module=jax.scipy)


@prtools.backend.numpy.overload
def broadcast_to(array, shape):
    # jax.numpy.broadcast_to expects `array` to be an array and fails when
    # passed a tuple
    return jax.numpy.broadcast_to(jax.numpy.asarray(array), shape)


@prtools.backend.numpy.overload
def dot(a, b, out=None):
    # jax.numpy.dot doesn't support the `out` parameter so we'll ignore it
    return jax.numpy.dot(a, b)


@prtools.backend.numpy.overload
def multiply(a, b, out=None):
    # jax.numpy.multiply doesn't support the `out` parameter so we'll
    # ignore it
    return jax.numpy.multiply(a, b)


@prtools.backend.numpy.overload
def divide(a, b, out=None):
    # jax.numpy.divide doesn't support the `out` parameter so we'll
    # ignore it
    return jax.numpy.divide(a, b)
