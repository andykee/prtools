import sys
import types


class Backend(types.ModuleType):
    _default_backend = 'numpy'
    _backend = None

    numpy = None
    scipy = None

    @classmethod
    def init_backend(cls):
        """Initialize the backend

        1) Registers the numpy and scipy backends
        2) Sets the backend to the default
        """
        cls.numpy = BackendManager(numpy=NumpyBackend,
                                   jax=JaxNumpyBackend)
        cls.scipy = BackendManager(numpy=ScipyBackend,
                                   jax=JaxScipyBackend)
        cls.set_backend(cls._default_backend)

    @classmethod
    def set_backend(cls, backend):
        """Change the backend

        Parameters
        ----------
        backend : str, {'numpy', or 'jax'}
            Name of the backend to load.

        """
        cls.numpy.set_backend(backend)
        cls._backend = backend

    @classmethod
    def get_backend(cls):
        """Return the *name* of the currently used backend

        Returns
        -------
        name : str

        """
        return cls._backend


class BackendManager:
    def __init__(self, **backends):
        self._available_backends = backends

    def set_backend(self, backend):
        self._backend = self._available_backends[backend]()

    def get_backend(self):
        return self._backend.name

    def __getattr__(self, name):
        return getattr(self._backend, name)


class BackendLibrary:
    def __getattr__(self, name):
        return getattr(self._backend, name)


class NumpyBackend(BackendLibrary):
    def __init__(self):
        import numpy
        self._backend = numpy


class ScipyBackend(BackendLibrary):
    def __init__(self):
        import scipy
        self._backend = scipy


class JaxNumpyBackend(BackendLibrary):
    def __init__(self):
        import jax.numpy
        self._backend = jax.numpy

    def broadcast_to(self, array, shape):
        # overload broadcast_to to always recast `array` as an array
        jnp_asarray = getattr(self._backend, 'asarray')
        jnp_broadcast_to = getattr(self._backend, 'broadcast_to')
        return jnp_broadcast_to(jnp_asarray(array), shape)

    def dot(self, a, b, out=None):
        # jax.numpy.dot doesn't support the `out` parameter so we'll ignore it
        jnp_dot = getattr(self._backend, 'dot')
        return jnp_dot(a, b)

    def multiply(self, a, b, out=None):
        # jax.numpy.multiply doesn't support the `out` parameter so we'll
        # ignore it
        jnp_multiply = getattr(self._backend, 'multiply')
        return jnp_multiply(a, b)

    def divide(self, a, b, out=None):
        # jax.numpy.divide doesn't support the `out` parameter so we'll
        # ignore it
        jnp_divide = getattr(self._backend, 'divide')
        return jnp_divide(a, b)


class JaxScipyBackend(BackendLibrary):
    def __init__(self):
        import jax.scipy
        self._backend = jax.scipy


Backend.init_backend()

# https://stackoverflow.com/a/72911884
sys.modules[__name__].__class__ = Backend
