import importlib

from ._base import BackendLibrary


class Numpy(BackendLibrary):
    def __init__(self):
        super().__init__(importlib.import_module('jax.numpy'))
        self.jax = importlib.import_module('jax')

    def broadcast_to(self, array, shape):
        # jax numpy.broadcast_to expects an array input
        array = self.module.asarray(array)
        return self.module.broadcast_to(array, shape)

    def divide(self, a, b, out=None):
        # jax.numpy.divide doesn't support the `out` parameter so we
        # ignore it
        return self.module.divide(a, b)

    def dot(self, a, b, out=None):
        # jax.numpy.dot doesn't support the `out` parameter so we ignore it
        return self.module.dot(a, b)

    def floor(self, x, *args, **kwargs):
        # jax numpy.floor expects a scalar or array input. It also doesn't
        # support the `out` parameter
        kwargs.pop('out', None)
        x = self.module.asarray(x)
        return self.module.floor(x, *args, **kwargs)

    def max(self, a, *args, **kwargs):
        # jax numpy.max expects an array input
        array = self.module.asarray(a)
        return self.module.max(array, *args, **kwargs)

    def multiply(self, a, b, out=None):
        # jax.numpy.multiply doesn't support the `out` parameter so we
        # ignore it
        return self.module.multiply(a, b)

    def sum(self, a, *args, **kwargs):
        kwargs.pop('out', None)
        a = self.module.asarray(a)
        return self.module.sum(a, *args, **kwargs)

    def take(self, a, indices, *args, **kwargs):
        # jax numpy.take expects an array input for a and indices
        a = self.module.asarray(a)
        indices = self.module.asarray(indices)
        return self.module.take(a, indices, *args, **kwargs)

    def _multi_dot_three(self, a, b, c, axes, out):
        # compute the matrix triple product
        #
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
            return self.module.linalg.multi_dot((a, b, c))
        else:
            iter_axis = _iter_axis(axes)
            return self.jax.vmap(self._multi_dot, in_axes=[None, iter_axis, None], out_axes=iter_axis)(a, b, c)

    def _multi_dot(self, a, b, c):
        # wrapper function to support vmap call signature
        return self.module.linalg.multi_dot((a, b, c))


def _iter_axis(axes):
    # NOTE: this function is purposely written in pure Python to avoid
    # dealing with mutability issues when __backend__ is JAX
    mask = [0, 1, 2]
    for ax in axes:
        mask[ax] = None
    return [ax for ax in mask if ax is not None][0]


class Scipy(BackendLibrary):
    def __init__(self):
        super().__init__(importlib.import_module('jax.scipy'))
