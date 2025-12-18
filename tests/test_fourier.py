import numpy as np
import pytest

import prtools
from .import BACKENDS


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('shape', ((10,10), (11,11), (10,11)))
def test_dft2(backend, shape):
    prtools.use(backend)
    m, n = shape
    f = np.random.rand(m, n) + 1j * np.random.rand(m, n)

    F_dft = prtools.dft2(f, [1/m, 1/n], unitary=False)
    F_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))

    assert np.allclose(F_dft, F_fft, atol=1e-5)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('axes', ((0,1), (1,2), (0,2)))
def test_dft2_cube(backend, axes):
    prtools.use(backend)
    shape = (10, 11, 12)
    m, n = np.take(shape, axes)
    f = np.random.uniform(size=shape) + 1j * np.random.uniform(size=shape)

    F_dft = prtools.dft2(f, [1/m, 1/n], axes=axes, unitary=False)
    F_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f), axes=axes))

    assert np.allclose(F_dft, F_fft, atol=5e-5)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('shape', ((10,10), (11,11), (10,11)))
def test_idft2(backend, shape):
    prtools.use(backend)
    m, n = shape
    f = np.random.rand(m, n) + 1j * np.random.rand(m, n)

    F_dft = prtools.idft2(f, [1/m, 1/n], unitary=False)
    F_fft = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(f)))

    assert np.allclose(F_dft, F_fft, atol=1e-5)


@pytest.mark.parametrize('backend', BACKENDS)
def test_dft2_unitary(backend):
    prtools.use(backend)
    n = 10
    f = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    f_power = np.sum(np.abs(f)**2)

    F = prtools.dft2(f, 1/n, unitary=True)
    F_power = np.sum(np.abs(F)**2)

    assert np.allclose(f_power, F_power)


@pytest.mark.parametrize('backend', BACKENDS)
def test_dft2_shift(backend):
    prtools.use(backend)
    n = 100
    shift = np.round(np.random.uniform(low=-25, high=25, size=2))
    f = np.ones((n, n))

    F = prtools.dft2(f, 1/n, shift=shift)

    (xc, yc) = (np.floor(n/2), np.floor(n/2))
    (x, y) = prtools.centroid(np.abs(F)**2)
    observed_shift = (x-xc, y-yc)
    assert np.array_equal(shift, observed_shift)


@pytest.mark.parametrize('backend', BACKENDS)
def test_dft2_rectangle(backend):
    prtools.use(backend)
    m, n = 10, 11
    f = np.random.rand(m, n) + 1j * np.random.rand(m, n)

    F_dft = prtools.dft2(f, [1/m, 1/n], unitary=False)
    F_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))

    assert np.allclose(F_dft, F_fft, atol=1e-5)
