from collections import namedtuple

import numpy as np


index = namedtuple("index", "row col shape nnz")


def spindex(m):
    """Create a sparse coordinate list (COO) index object.

    Parameters
    ----------
    m : array_like
        Dense matrix to be vectorized

    Returns
    -------
    index
        namedtuple with the following attributes:

        * ``row`` List of row indices which contain nonzero data
        * ``col`` List of column indices which contain nonzero data
        * ``shape`` Tuple of dimensions of ``m``
        * ``nnz`` Number of nonzero entries in ``m``

    See Also
    --------
    * :func:`~prtools.spmatrix` Create a dense matrix from a sparse array
    * :func:`~prtools.sparray` Create a sparse array from a dense matrix
    """

    m = np.asarray(m)
    r, c = m.nonzero()
    shape = m.shape
    nnz = len(r)
    return index(row=r, col=c, shape=shape, nnz=nnz)


def spmask(index):
    """Create a mask from a sparse coordinate list (COO) index."""
    return spmatrix(np.ones(index.row.shape), index)


def spmatrix(a, index):
    """Create a dense matrix from a sparse array
    
    Parameters
    ----------
    a : array_like
        Sparse array to be reformed as a dense matrix
    index : index
        Corresponding index object

    Returns
    -------
    ndarray
        Dense matrix

    See Also
    --------
    * :func:`~prtools.sparray` Create a sparse array from a dense matrix
    * :func:`~prtools.spindex` Create a sparse coordinate list (COO) index
      object
    """

    m = np.zeros(index.shape)
    for n in range(index.nnz):
        m[index.row[n], index.col[n]] = a[n]
    return m



def sparray(m, index=None):
    """Create a sparse array from a dense matrix

    Parameters
    ----------
    m : array_like
        Dense matrix to be reformed as a sparse vector
    index : index
        Corresponding index object

    Returns
    -------
    ndarray
        Sparse vector
    
    See Also
    --------
    * :func:`~prtools.spmatrix` Create a dense matrix from a sparse array
    * :func:`~prtools.spindex` Create a sparse coordinate list (COO) index
      object
    """

    if index is None:
        index = spindex(m)

    rmi = np.ravel_multi_index((index.row, index.col),
                              index.shape, order='C')
    a = m.ravel()
    return a[rmi]
    
