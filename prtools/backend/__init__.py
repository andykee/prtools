import importlib


class BackendRegistry:

    _BUILTIN_BACKEND = ['numpy', 'jax']

    def _get_backend_module(self, backend):
        return importlib.import_module(f'.{backend}', 'prtools.backend')

    def load_numpy(self, backend):
        backend = self._get_backend_module(backend)
        return backend.Numpy()

    def load_scipy(self, backend):
        return self._get_backend_module(backend).Scipy()


registry = BackendRegistry()
