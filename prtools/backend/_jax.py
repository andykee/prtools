import importlib

from ._base import BackendLibrary


class Numpy(BackendLibrary):
    def __init__(self):
        super().__init__(importlib.import_module('jax.numpy'))

    def broadcast_to(self, array, shape):
        # jax broadcast_to expects an array input
        array = self.module.asarray(array)
        return self.module.broadcast_to(array, shape)

    def dot(self, a, b, out=None):
        # jax.numpy.dot doesn't support the `out` parameter so we ignore it
        return self.module.dot(a, b)

    def max(self, a, *args, **kwargs):
        # jax max expects an array input
        array = self.module.asarray(a)
        return self.module.max(array, *args, **kwargs)

    def multiply(self, a, b, out=None):
        # jax.numpy.multiply doesn't support the `out` parameter so we
        # ignore it
        return self.module.multiply(a, b)

    def divide(self, a, b, out=None):
        # jax.numpy.divide doesn't support the `out` parameter so we
        # ignore it
        return self.module.divide(a, b)


class Scipy(BackendLibrary):
    def __init__(self):
        super().__init__(importlib.import_module('jax.scipy'))
