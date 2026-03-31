import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import prtools
from . import BACKENDS


@pytest.mark.parametrize('xp', BACKENDS)
def test_centroid(xp):
    x = np.zeros((5, 5))
    x[2, 2] = 1
    x = xp.asarray(x)
    assert xp.array_equal(prtools.centroid(x), [2, 2])


@pytest.mark.parametrize('xp', BACKENDS)
def test_centroid_where(xp):
    x = np.zeros((5, 5))
    x[2, 2] = 1
    x[1, 1] = 1
    m = np.ones_like(x)
    m[1, 1] = 0
    x = xp.asarray(x)
    assert xp.array_equal(prtools.centroid(x, where=m), [2, 2])


@pytest.mark.parametrize('xp', BACKENDS)
def test_centroid_nan(xp):
    x = np.zeros((5, 5))
    x[2, 2] = 1
    x[2, 3] = np.nan
    x = xp.asarray(x)
    assert xp.array_equal(prtools.centroid(x), [2, 2])


@pytest.mark.parametrize('xp', BACKENDS)
def test_medfix(xp):
    x, _ = np.meshgrid(range(10), range(10))
    x[2, 2] = 100
    m = np.zeros_like(x)
    m[2, 2] = 1
    x = xp.asarray(x)
    y = prtools.medfix(x, mask=m, kernel=(3, 3))
    assert y[2, 2] == 2


@pytest.mark.parametrize('xp', BACKENDS)
def test_medfix_bigmask(xp):
    x, _ = np.meshgrid(range(10), range(10))
    x[2, 2] = 100
    m = np.zeros_like(x)
    m[2:6, 2:6] = 1
    x = xp.asarray(x)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        y = prtools.medfix(x, mask=m, kernel=(3, 3))
    assert xp.all(xp.isnan(y[3:5, 3:5]))


@pytest.mark.parametrize('xp', BACKENDS)
def test_boundary(xp):
    x = np.zeros((10, 10))
    x[3:7, 2:8] = 1
    x = xp.asarray(x)
    assert xp.array_equal(prtools.boundary(x), (3, 6, 2, 7))


@pytest.mark.parametrize('xp', BACKENDS)
def test_rebin(xp):
    x = xp.ones((10, 10))
    assert xp.array_equal(prtools.rebin(x, 2), 4*xp.ones((5, 5)))


@pytest.mark.parametrize('xp', BACKENDS)
def test_ndrebin(xp):
    x = xp.ones((3, 10, 10))
    assert xp.array_equal(prtools.rebin(x, 2), 4*xp.ones((3, 5, 5)))


@pytest.mark.parametrize('xp', BACKENDS)
def test_shift(xp):
    x1 = np.zeros((5,5))
    x1[2,2] = 1

    xs = np.zeros((5,5))
    xs[3,3] = 1

    x1 = xp.asarray(x1)
    xs = xp.asarray(xs)

    x2 = prtools.shift(x1, (1,1))

    assert(xp.allclose(x2, xs, atol=1e-6))


@pytest.mark.parametrize('xp', BACKENDS)
def test_register(xp):
    x1 = np.zeros((5,5))
    x1[2,2] = 1

    x2 = np.zeros((5,5))
    x2[3,3] = 1

    x1 = xp.asarray(x1)
    x2 = xp.asarray(x2)

    s = prtools.register(x2, x1, oversample=2)

    assert(xp.allclose(xp.asarray(s), xp.asarray((-1,-1))))