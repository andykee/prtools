import numpy as np
import pytest

import prtools


def phi1(stil, shat, w):
    # Eqs. 3 and 4
    out = []
    for s, sh in zip(stil, shat):
        alpha = np.sum(w*s*sh)/np.sum(w*sh**2)
        out.append(np.sum(w*(alpha*sh - s)**2)/np.sum(w*s**2))
    return np.mean(out)


def phi2(stil, shat, w):
    # Eqs. 11-14
    out = []
    for s, sh in zip(stil, shat):
        gtil = s - np.sum(w*s)/np.sum(w)
        ghat = sh - np.sum(w*sh)/np.sum(w)
        alpha = np.sum(w*gtil*ghat)/np.sum(w*ghat**2)
        out.append(np.sum(w*(alpha*ghat - gtil)**2)/np.sum(w*s**2))
    return np.mean(out)


def taper(shape):
    # Smooth weighting function similar to Eq. 8
    r = np.hypot(*prtools.mesh(shape))
    return np.where(r < 8, 1.0,
                    np.where(r <= 12, np.cos(np.pi*(r - 8)/8)**2, 0.0))


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    data = rng.random((3, 32, 32)) * np.array([1, 10, 100])[:, None, None]
    est = data + 0.05 * rng.standard_normal(data.shape) * data.mean()
    return data, est


@pytest.mark.parametrize('weight', ['none', 'bool', 'taper'])
@pytest.mark.parametrize('gain_bias_invariant, ref', [(False, phi1),
                                                      (True, phi2)])
def test_sserror_matches_paper(frames, weight, gain_bias_invariant, ref):
    data, est = frames
    if weight == 'none':
        mask, w = None, np.ones(data.shape[-2:])
    elif weight == 'bool':
        mask = taper(data.shape[-2:]) > 0
        w = mask.astype(float)
    else:
        mask = w = taper(data.shape[-2:])

    err = prtools.sserror(data, est, mask=mask,
                          gain_bias_invariant=gain_bias_invariant)
    assert np.allclose(err, ref(data, est, w))


@pytest.mark.parametrize('gain_bias_invariant', [False, True])
def test_sserror_2d_matches_single_plane(frames, gain_bias_invariant):
    data, est = frames
    err2d = prtools.sserror(data[0], est[0],
                            gain_bias_invariant=gain_bias_invariant)
    err3d = prtools.sserror(data[:1], est[:1],
                            gain_bias_invariant=gain_bias_invariant)
    assert np.allclose(err2d, err3d)


@pytest.mark.parametrize('gain_bias_invariant', [False, True])
def test_sserror_identical_is_zero(frames, gain_bias_invariant):
    data, _ = frames
    assert np.allclose(prtools.sserror(data, data,
                       gain_bias_invariant=gain_bias_invariant), 0)


@pytest.mark.parametrize('gain_bias_invariant', [False, True])
def test_sserror_gain_invariant(frames, gain_bias_invariant):
    data, _ = frames
    gain = np.array([0.5, 2, 7])[:, None, None]
    assert np.allclose(prtools.sserror(data, gain*data,
                       gain_bias_invariant=gain_bias_invariant), 0)


def test_sserror_phi1_is_not_bias_invariant(frames):
    data, _ = frames
    assert prtools.sserror(data, data + 5) > 1e-3


def test_sserror_phi2_is_gain_bias_invariant(frames):
    data, _ = frames
    gain = np.array([0.5, 2, 7])[:, None, None]
    bias = np.array([-3, 5, 40])[:, None, None]
    assert np.allclose(prtools.sserror(data, gain*data + bias,
                                       gain_bias_invariant=True), 0)


def test_sserror_ghat(frames):
    data, est = frames
    ghat = est - est.mean(axis=(-2, -1), keepdims=True)
    assert np.allclose(
        prtools.sserror(data, est, gain_bias_invariant=True),
        prtools.sserror(data, est, gain_bias_invariant=True, ghat=ghat))
