import numpy as np
import pytest

import prtools
from .import BACKENDS


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('shape', ((255,255), (256,256)))
def test_gauss_kernel(backend, shape):
    prtools.use(backend)

    # note we need to use a sufficiently large sigma here to avoid wrapping
    # in the FFT-computed version of the kernel
    sigma = 5
    #shape = (256,256)
    G = prtools.gauss_kernel(shape, sigma, fftshift=True)

    r = np.arange(-shape[0]//2,shape[0]//2)
    c = np.arange(-shape[1]//2,shape[1]//2)
    g = prtools.gauss(r, c, sigma)
    G_fft = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g))))
    G_fft = G_fft / np.max(G_fft)
    
    assert np.allclose(G, G_fft, atol=1e-6)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('shape', ((255,255), (256,256)))
def test_gauss_kernel_pixelscale(backend, shape):
    prtools.use(backend)

    r = np.linspace(-10, 10, shape[0])
    c = np.linspace(-10, 10, shape[1])
    nr, nc = len(r), len(c)
    dr, dc = r[1] - r[0], c[1] - c[0]

    G = prtools.gauss_kernel((nr, nc), sigma=1, pixelscale=(dr, dc),
                             fftshift=True)

    g = prtools.gauss(r, c, sigma=1)
    G_fft = np.abs(np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(g))))
    G_fft = G_fft / np.max(G_fft)
    
    assert np.allclose(G, G_fft, atol=1e-6)


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('shape', ((255,255), (256,256)))
def test_gauss_kernel_oversample(backend, shape):
    prtools.use(backend)

    sigma = 5
    oversample = 3

    G = prtools.gauss_kernel(shape, sigma=sigma, fftshift=False)
    Go = prtools.gauss_kernel((shape[0]*oversample, shape[1]*oversample),
                              sigma=sigma, oversample=oversample,
                              fftshift=False)
    
    assert np.allclose(G[0:100,0:100], Go[0:100,0:100])


@pytest.mark.parametrize('backend', BACKENDS)
@pytest.mark.parametrize('shape', ((255,255), (256,256)))
def test_pixel_kernel_oversample(backend, shape):
    prtools.use(backend)

    oversample = 3

    P = prtools.pixel_kernel(shape, fftshift=False)
    Po = prtools.pixel_kernel((shape[0]*oversample, shape[1]*oversample),
                              oversample=oversample, fftshift=False)
    
    assert np.allclose(P[0:100,0:100], Po[0:100,0:100])