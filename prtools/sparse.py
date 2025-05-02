import numpy as np


class index:
    """Sparse coordinate list (COO) index

    Parameters
    ----------
    row : array_like
        List of row indices which contain nonzero data
    col : array_like
        List of column indices which contain nonzero data
    shape : (2,) array_like
        Dimensions of dense matrix

    Attributes
    ----------
    nnz : int
        Number of nonzero entries in dense matrix

    Returns
    -------
    index

    See Also
    --------
    * :func:`~prtools.sparse` Create a sparse array from a dense matrix
    * :func:`~prtools.dense` Create a dense matrix from a sparse array
    """

    def __init__(self, row, col, shape):
        self.row = row
        self.col = col
        self.shape = shape
        self.nnz = len(self.row)


def dense(a, index):
    """Create a dense matrix from a sparse array

    Parameters
    ----------
    a : array_like
        Sparse array to be reformed as a dense matrix
    index : :class:`~prtools.index`
        Corresponding index object

    Returns
    -------
    ndarray
        Dense matrix

    See Also
    --------
    * :func:`~prtools.sparse` Create a sparse array from a dense matrix
    * :func:`~prtools.index` Create a sparse coordinate list (COO) index
      object
    """

    m = np.zeros(index.shape)
    for n in range(index.nnz):
        m[index.row[n], index.col[n]] = a[n]
    return m


def sparse(m, index=None):
    """Create a sparse array from a dense matrix

    Parameters
    ----------
    m : array_like
        Dense matrix to be reformed as a sparse vector
    index : :class:`~prtools.index`
        Corresponding index object

    Returns
    -------
    ndarray
        Sparse vector

    See Also
    --------
    * :func:`~prtools.dense` Create a dense matrix from a sparse array
    * :func:`~prtools.index` Create a sparse coordinate list (COO) index
      object
    """

    if index is None:
        index = index(m)

    rmi = np.ravel_multi_index((index.row, index.col),
                               index.shape, order='C')
    a = m.ravel()
    return a[rmi]


def index_from_mask(m):
    """Create a sparse coordinate list (COO) index object from a
    mask.

    Parameters
    ----------
    m : array_like
        Dense matrix to be vectorized

    Returns
    -------
    :class:`~prtools.index`
    """

    m = np.asarray(m)
    r, c = m.nonzero()
    shape = m.shape
    return index(row=r, col=c, shape=shape)


def mask_from_index(index):
    """Create a mask from a sparse coordinate list (COO) index.

    Parameters
    ----------
    index : :class:`~prtools.index`
        Index object

    Returns
    -------
    ndarray
        Dense matrix
    """

    return dense(np.ones(index.row.shape), index)
