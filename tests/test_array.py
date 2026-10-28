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
@pytest.mark.parametrize('mode, indexing, result', [
    ('absolute', 'ij', (44, 94)),
    ('absolute', 'xy', (94, 44)),
    ('center', 'ij', (-20, 30)),
    ('center', 'xy', (30, -20)),
])
def test_centroid_mode_indexing(xp, mode, indexing, result):
    # circle shifted 20 rows up and 30 columns right of center
    circ = prtools.circle((128, 128), 25, shift=(-20, 30), antialias=False)
    c = prtools.centroid(xp.asarray(circ), mode=mode, indexing=indexing)
    assert np.allclose(np.asarray(c), result)


@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('n', [64, 65])
@pytest.mark.parametrize('indexing', ['ij', 'xy'])
def test_centroid_center_mode_centered(xp, n, indexing):
    # center mode measures from the mesh origin at n//2, for odd n too
    circ = prtools.circle((n, n), 20, antialias=False)
    c = prtools.centroid(xp.asarray(circ), mode='center', indexing=indexing)
    assert np.allclose(np.asarray(c), (0, 0))


@pytest.mark.parametrize('mode, shift_ij, shift_xy, center', [
    ('center', (5, -7), (-7, 5), (37, 33)),
    ('absolute', (30, 20), (20, 30), (30, 20)),
])
def test_crop_indexing(mode, shift_ij, shift_xy, center):
    x = np.arange(64*80).reshape(64, 80)
    ij = prtools.crop(x, (10, 12), shift=shift_ij, mode=mode, indexing='ij')
    xy = prtools.crop(x, (10, 12), shift=shift_xy, mode=mode, indexing='xy')
    assert np.array_equal(ij, xy)
    assert ij[5, 6] == x[center]


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

    assert xp.allclose(xp.asarray(s), xp.asarray((-1, -1)))


@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('shape, oversample, offset, result', [
    ((2, 2), 1, 0, [[0, 1], [2, 3]]),
    ((2, 2), 2, 0, [3]),
    ((2, 2), 2, (-1, 0), [1]),
    ((2, 2), 2, (0, -1), [2]),
    ((2, 2), 2, (-1, -1), [0]),
    ((3, 3), 1, 0, [[0, 1, 2], [3, 4, 5], [6, 7, 8]]),
    ((3, 3), 3, 0, [4]),
    ((3, 3), 3, (-1, 0), [1]),
    ((3, 3), 3, (1, 0), [7]),
    ((3, 3), 3, (0, -1), [3]),
    ((3, 3), 3, (0, 1), [5]),
    ((3, 3), 3, (-1, -1), [0]),
    ((3, 3), 3, (-1, 1), [2]),
    ((3, 3), 3, (1, -1), [6]),
    ((3, 3), 3, (1, 1), [8]),
    ((4, 4), 1, 0, [[0, 1, 2, 3], [4, 5, 6, 7],
                    [8, 9, 10, 11], [12, 13, 14, 15]]),
    ((4, 4), 2, 0, [[5, 7], [13, 15]]),
    ])
def test_sample(xp, shape, oversample, offset, result):
    x = np.arange(np.prod(shape)).reshape(shape)
    assert np.all(prtools.sample(x, oversample, offset) == result)

@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('shape, oversample, offset', [
    ((2, 2), 1, 1),
    ((2, 2), 1, -2),
    ((2, 2), 2, 1),
    ((3, 3), 3, -2),
    ((3, 3), 3, 2),
    ((4, 4), 2, 1),
    ((4, 4), 2, -2)
    ])
def test_sample_value_error(xp, shape, oversample, offset):
    x = np.arange(np.prod(shape)).reshape(shape)
    with pytest.raises(ValueError):
        prtools.sample(x, oversample, offset)


@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('oversample, offset_ij, offset_xy, result', [
    (3, (1, 0), (0, 1), [7]),       # +row is +y
    (3, (0, -1), (-1, 0), [3]),     # -col is -x
    (3, (-1, 1), (1, -1), [2]),
    ((3, 1), (1, 0), (0, 1), [[6, 7, 8]]),
])
def test_sample_indexing(xp, oversample, offset_ij, offset_xy, result):
    # xy reverses the order of offset (and oversample) but not their signs
    x = np.arange(9).reshape(3, 3)
    ij = prtools.sample(x, oversample, offset_ij, indexing='ij')
    if np.ndim(oversample):
        oversample = oversample[::-1]
    xy = prtools.sample(x, oversample, offset_xy, indexing='xy')
    assert np.all(ij == result)
    assert np.all(xy == result)
