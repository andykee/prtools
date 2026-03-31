import jax.numpy as jnp
import numpy as np
import pytest

import prtools
from . import BACKENDS


@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('shape', ((10,10), (11,11), (10,11)))
def test_dft2(xp, shape):
    m, n = shape
    f = np.random.rand(m, n) + 1j * np.random.rand(m, n)

    f = xp.asarray(f)

    F_dft = prtools.dft2(f, [1/m, 1/n], unitary=False)
    F_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(f)))

    assert xp.allclose(F_dft, F_fft, atol=1e-5)


@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('axes', ((0,1), (1,2), (0,2)))
def test_dft2_cube(xp, axes):
    shape = (10, 11, 12)
    m, n = np.take(shape, axes)
    f = np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)
    f - xp.asarray(f)

    F_dft = prtools.dft2(f, [1/m, 1/n], axes=axes, unitary=False)
    F_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(f), axes=axes))

    assert xp.allclose(F_dft, F_fft, atol=5e-5)


@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('shape', ((10,10), (11,11), (10,11)))
def test_idft2(xp, shape):
    m, n = shape
    f = np.random.rand(m, n) + 1j * np.random.rand(m, n)
    f - xp.asarray(f)

    F_dft = prtools.idft2(f, [1/m, 1/n], unitary=False)
    F_fft = xp.fft.fftshift(xp.fft.ifft2(xp.fft.ifftshift(f)))

    assert xp.allclose(F_dft, F_fft, atol=1e-5)


@pytest.mark.parametrize('xp', BACKENDS)
def test_dft2_unitary(xp):
    n = 10
    f = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    f = xp.asarray(f)

    f_power = xp.sum(xp.abs(f)**2)

    F = prtools.dft2(f, 1/n, unitary=True)
    F_power = xp.sum(xp.abs(F)**2)

    assert xp.allclose(f_power, F_power)


@pytest.mark.parametrize('xp', BACKENDS)
def test_dft2_shift(xp):
    n = 100
    shift = np.round(np.random.uniform(low=-25, high=25, size=2))
    f = xp.ones((n, n))

    F = prtools.dft2(f, 1/n, shift=shift)

    (xc, yc) = (np.floor(n/2), np.floor(n/2))
    (x, y) = prtools.centroid(xp.abs(F)**2)
    observed_shift = (x-xc, y-yc)
    assert xp.array_equal(shift, observed_shift)


@pytest.mark.parametrize('xp', BACKENDS)
def test_dft2_rectangle(xp):
    m, n = 10, 11
    f = np.random.rand(m, n) + 1j * np.random.rand(m, n)
    f = xp.asarray(f)

    F_dft = prtools.dft2(f, [1/m, 1/n], unitary=False)
    F_fft = xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(f)))

    assert xp.allclose(F_dft, F_fft, atol=1e-5)
