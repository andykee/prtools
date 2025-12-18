from ._base import BackendLibrary


class Numpy(BackendLibrary):
    def __init__(self):
        import numpy
        super().__init__(numpy)

    def _multi_dot_three(self, a, b, c, axes, out):
        # compute the matrix triple product
        #
        # a few notes:
        # * this method is similar to np.linalg.multi_dot although it is less
        #   general - here we only consider the matrix triple product used as
        #   a part of prtools.dft2
        # * because we use np.matmul instead of np.linalg.multi_dot, we can
        #   take advantage of broadcasting a and c when b.ndim = 3. This
        #   eliminates a for loop in the code
        # * the implementation used here supports b with ndim in (2, 3)
        #   iterating over any of the 3 axes when b.ndim == 3
        # * the implementation used here is actually slightly faster than
        #   an equivalent call to np.linalg.multi_dot when b.ndim == 2
        # * np.linalg.multi_dot chooses the fastest multiplication order from
        #   [(ab)c, a(bc)] depending on the shapes of a, b, and c. There is no
        #   difference when computing the dft because both a and c are square
        #   matrices
        ab = self.module.matmul(a, b, axes=[(0, 1), axes, axes])
        out = self.module.matmul(ab, c, axes=[axes, (0, 1), axes], out=out)
        return out


class Scipy(BackendLibrary):
    def __init__(self):
        import scipy
        super().__init__(scipy)
