from ._base import _BackendLibrary


class Numpy(_BackendLibrary):
    def __init__(self):
        import numpy
        super().__init__(numpy)


class Scipy(_BackendLibrary):
    def __init__(self):
        import scipy
        super().__init__(scipy)
