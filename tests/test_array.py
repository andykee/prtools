import warnings

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


x, _ = np.meshgrid(range(10), range(10))
x[2,2] = 100

def test_medfix():
    m = np.zeros_like(x)
    m[2,2] = 1
    y = prtools.medfix(x, mask=m, kernel=(3,3))
    assert(y[2,2] == 2)

def test_medfix_bigmask():
    m = np.zeros_like(x)
    m[2:6, 2:6] = 1
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = prtools.medfix(x, mask=m, kernel=(3,3))
    assert(np.all(np.isnan(y[3:5,3:5])))
