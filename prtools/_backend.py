import sys
import types

from prtools.backend import registry
from prtools.backend._base import _BackendLibrary


class _BackendName:
    """Helper class for storing the backend name - needed since strings are
    immutable.
    """

    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name

    def __eq__(self, other):
        return self.name == other


class _Backend(types.ModuleType):
    __backend__ = _BackendName(None)
    numpy = _BackendLibrary(None)
    scipy = _BackendLibrary(None)

    @classmethod
    def use(cls, backend):
        """Select the backend used for N-dimensional array operations.

        Parameters
        ----------
        backend : str
            The backend to use. This can be any of the following backends,
            which are case-insensitive:
            
            * NumPy
            * JAX
        """
        #if name in registry:
        backend = backend.lower()
        cls.__backend__.name = backend
        cls.numpy.module = registry.load_numpy(backend)
        cls.scipy.module = registry.load_scipy(backend)


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


# use NumPy by default
_Backend.use('numpy')

# https://stackoverflow.com/a/72911884
sys.modules[__name__].__class__ = _Backend
