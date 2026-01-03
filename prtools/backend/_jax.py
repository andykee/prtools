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


class Scipy(BackendLibrary):
    def __init__(self):
        super().__init__(importlib.import_module('jax.scipy'))
