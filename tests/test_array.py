import numpy as np

import prtools


def test_centroid():
    x = np.zeros((5,5))
    x[2,2] = 1
    assert(np.array_equal(prtools.centroid(x), [2,2]))

def test_centroid_where():
    x = np.zeros((5,5))
    x[2,2] = 1
    x[1,1] = 1
    mask = np.ones_like(x)
    mask[1,1] = 0
    assert(np.array_equal(prtools.centroid(x, where=mask), [2,2]))

def test_centroid_nan():
    x = np.zeros((5,5))
    x[2,2] = 1
    x[2,3] = np.nan
    assert(np.array_equal(prtools.centroid(x), [2,2]))

