import jax.numpy as jnp
import numpy as np
import pytest

import prtools
from . import BACKENDS


@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('shape', ((255,255), (256,256)))
def test_gauss_kernel(xp, shape):
    # note we need to use a sufficiently large sigma here to avoid wrapping
    # in the FFT-computed version of the kernel
    sigma = 5
    #shape = (256,256)
    G = prtools.gauss_kernel(shape, sigma, fftshift=True, xp=xp)

    r = np.arange(-shape[0]//2,shape[0]//2)
    c = np.arange(-shape[1]//2,shape[1]//2)
    g = prtools.gauss(r, c, sigma, xp=xp)
    G_fft = xp.abs(xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(g))))
    G_fft = G_fft / xp.max(G_fft)
    
    assert xp.allclose(G, G_fft, atol=1e-6)


@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('shape', ((255,255), (256,256)))
def test_gauss_kernel_pixelscale(xp, shape):
    r = np.linspace(-10, 10, shape[0])
    c = np.linspace(-10, 10, shape[1])
    nr, nc = len(r), len(c)
    dr, dc = r[1] - r[0], c[1] - c[0]

    G = prtools.gauss_kernel((nr, nc), sigma=1, pixelscale=(dr, dc),
                             fftshift=True, xp=xp)

    g = prtools.gauss(r, c, sigma=1, xp=xp)
    G_fft = xp.abs(xp.fft.fftshift(xp.fft.fft2(xp.fft.ifftshift(g))))
    G_fft = G_fft / xp.max(G_fft)
    
    assert xp.allclose(G, G_fft, atol=1e-6)


@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('shape', ((255,255), (256,256)))
def test_gauss_kernel_oversample(xp, shape):
    sigma = 5
    oversample = 3

    G = prtools.gauss_kernel(shape, sigma=sigma, fftshift=False, xp=xp)
    Go = prtools.gauss_kernel((shape[0]*oversample, shape[1]*oversample),
                              sigma=sigma, oversample=oversample,
                              fftshift=False, xp=xp)
    
    assert xp.allclose(G[0:100,0:100], Go[0:100,0:100])


@pytest.mark.parametrize('xp', BACKENDS)
@pytest.mark.parametrize('shape', ((255,255), (256,256)))
def test_pixel_kernel_oversample(xp, shape):
    oversample = 3

    P = prtools.pixel_kernel(shape, fftshift=False, xp=xp)
    Po = prtools.pixel_kernel((shape[0]*oversample, shape[1]*oversample),
                              oversample=oversample, fftshift=False, xp=xp)
    
    assert xp.allclose(P[0:100,0:100], Po[0:100,0:100])