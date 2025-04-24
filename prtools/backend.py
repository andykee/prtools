import sys
import types

import numpy as _numpy
import scipy as _scipy


class BackendLibrary:
    def __init__(self, module):
        self.module = module

    def __getattr__(self, name):
        return getattr(self.module, name)

    def overload(self, fun):
        """A decorator for overloading backend functionality"""
        setattr(self, fun.__name__, fun)


class BackendComparator:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other


class Backend(types.ModuleType):
    __backend__ = BackendComparator('numpy')
    numpy = BackendLibrary(_numpy)
    scipy = BackendLibrary(_scipy)

    @classmethod
    def set_backend(cls, name, numpy_module, scipy_module):
        """Change the backend

        Parameters
        ----------
        name : str
            Name of the backend.
        numpy_module : module
            Library providing numpy-like functionality
        scipy_module : module
            Library providing scipy-like functionality
        """
        cls.__backend__.name = name
        cls.numpy.module = numpy_module
        cls.scipy.module = scipy_module




#    def broadcast_to(self, array, shape):
#        # overload broadcast_to to always recast `array` as an array
#        jnp_asarray = getattr(self._backend, 'asarray')
#        jnp_broadcast_to = getattr(self._backend, 'broadcast_to')
#        return jnp_broadcast_to(jnp_asarray(array), shape)
#
#    def dot(self, a, b, out=None):
#        # jax.numpy.dot doesn't support the `out` parameter so we'll ignore it
#        jnp_dot = getattr(self._backend, 'dot')
#        return jnp_dot(a, b)
#
#    def multiply(self, a, b, out=None):
#        # jax.numpy.multiply doesn't support the `out` parameter so we'll
#        # ignore it
#        jnp_multiply = getattr(self._backend, 'multiply')
#        return jnp_multiply(a, b)
#
#    def divide(self, a, b, out=None):
#        # jax.numpy.divide doesn't support the `out` parameter so we'll
#        # ignore it
#        jnp_divide = getattr(self._backend, 'divide')
#        return jnp_divide(a, b)


# Initialize the backend to use numpy/scipy by default
#Backend.set_backend('numpy', numpy_lib=numpy, scipy_lib=scipy)

#Backend.initialize()

# https://stackoverflow.com/a/72911884
sys.modules[__name__].__class__ = Backend
