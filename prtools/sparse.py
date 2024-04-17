import numpy as np


class spindex:
    """Sparse coordinate list (COO) index

    Parameters
    ----------
    row : array_like
        List of row indices which contain nonzero data
    col : array_like
        List of column indices which contain nonzero data
    shape : (2,) array_like
        Dimensions of dense matrix
    nnz : int, optional
        Number of nonzero entries in dense matrix. If not 
        specified, 
    Returns
    -------
    spindex

    See Also
    --------
    * :func:`~prtools.sparray` Create a sparse array from a dense matrix
    * :func:`~prtools.spmatrix` Create a dense matrix from a sparse array
    """

    def __init__(self, row, col, shape, nnz=None):
        self.row = row
        self.col = col
        self.shape = shape
        self.nnz = nnz if nnz is not None else len(self.row)


def spmatrix(a, index):
    """Create a dense matrix from a sparse array
    
    Parameters
    ----------
    a : array_like
        Sparse array to be reformed as a dense matrix
    index : :class:`~prtools.spindex`
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
    index : :class:`~prtools.spindex`
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
    

def spindex_from_mask(m):
    """Create a sparse coordinate list (COO) index object from a
    mask.

    Parameters
    ----------
    m : array_like
        Dense matrix to be vectorized

    Returns
    -------
    :class:`~prtools.spindex`
    """

    m = np.asarray(m)
    r, c = m.nonzero()
    shape = m.shape
    nnz = len(r)
    return spindex(row=r, col=c, shape=shape, nnz=nnz)


def mask_from_spindex(index):
    """Create a mask from a sparse coordinate list (COO) index.
    
    Parameters
    ----------
    index : :class:`~prtools.spindex`
        Index object

    Returns
    -------
    ndarray
        Dense matrix
    """

    return spmatrix(np.ones(index.row.shape), index)
