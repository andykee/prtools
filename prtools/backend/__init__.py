import importlib
import sys
import types

from ._base import BackendLibrary


def backend_available(name):
    """Return True if backend library is available"""
    try:
        importlib.import_module(name)
    except ImportError:
        return False
    else:
        return True
        

class BackendName:
    """Helper class for changing the backend name - needed since strings are
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


class BackendRegistry:

    _BUILTIN_BACKEND = ['numpy', 'jax']

    def _get_backend_module(self, backend):
        return importlib.import_module(f'._{backend}', 'prtools.backend')

    def load_numpy(self, backend):
        backend = self._get_backend_module(backend)
        return backend.Numpy()

    def load_scipy(self, backend):
        return self._get_backend_module(backend).Scipy()


registry = BackendRegistry()

class Backend(types.ModuleType):
    
    numpy = BackendLibrary(None)
    scipy = BackendLibrary(None)

    __backend__ = BackendName(None)

    JAX_AVAILABLE = backend_available('jax') & backend_available('optax')
    NUMPY_AVAILABLE = backend_available('numpy')

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
Backend.use('numpy')

# https://stackoverflow.com/a/72911884
sys.modules[__name__].__class__ = Backend
