import warnings

import numpy as np
import pytest

import prtools
from .import BACKENDS


@pytest.mark.parametrize('backend', BACKENDS)
def test_centroid(backend):
    prtools.use(backend)
    x = np.zeros((5, 5))
    x[2, 2] = 1
    assert np.array_equal(prtools.centroid(x), [2, 2])


@pytest.mark.parametrize('backend', BACKENDS)
def test_centroid_where(backend):
    prtools.use(backend)
    x = np.zeros((5, 5))
    x[2, 2] = 1
    x[1, 1] = 1
    mask = np.ones_like(x)
    mask[1, 1] = 0
    assert np.array_equal(prtools.centroid(x, where=mask), [2, 2])


@pytest.mark.parametrize('backend', BACKENDS)
def test_centroid_nan(backend):
    prtools.use(backend)
    x = np.zeros((5, 5))
    x[2, 2] = 1
    x[2, 3] = np.nan
    assert np.array_equal(prtools.centroid(x), [2, 2])


x, _ = np.meshgrid(range(10), range(10))
x[2, 2] = 100


@pytest.mark.parametrize('backend', BACKENDS)
def test_medfix(backend):
    prtools.use(backend)
    m = np.zeros_like(x)
    m[2, 2] = 1
    y = prtools.medfix(x, mask=m, kernel=(3, 3))
    assert y[2, 2] == 2


@pytest.mark.parametrize('backend', BACKENDS)
def test_medfix_bigmask(backend):
    prtools.use(backend)
    m = np.zeros_like(x)
    m[2:6, 2:6] = 1
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = prtools.medfix(x, mask=m, kernel=(3, 3))
    assert np.all(np.isnan(y[3:5, 3:5]))


@pytest.mark.parametrize('backend', BACKENDS)
def test_boundary(backend):
    prtools.use(backend)
    x = np.zeros((10, 10))
    x[3:7, 2:8] = 1
    assert np.array_equal(prtools.boundary(x), (3, 6, 2, 7))


@pytest.mark.parametrize('backend', BACKENDS)
def test_rebin(backend):
    prtools.use(backend)
    x = np.ones((10, 10))
    assert np.array_equal(prtools.rebin(x, 2), 4*np.ones((5, 5)))


@pytest.mark.parametrize('backend', BACKENDS)
def test_ndrebin(backend):
    prtools.use(backend)
    x = np.ones((3, 10, 10))
    assert np.array_equal(prtools.rebin(x, 2), 4*np.ones((3, 5, 5)))