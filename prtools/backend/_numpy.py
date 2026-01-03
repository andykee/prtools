from ._base import BackendLibrary


class Numpy(BackendLibrary):
    def __init__(self):
        import numpy
        super().__init__(numpy)


class Scipy(BackendLibrary):
    def __init__(self):
        import scipy
        super().__init__(scipy)
