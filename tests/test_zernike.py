import warnings

import numpy as np
import pytest

import prtools
from prtools.zernike import R, noll_to_nm


def test_zernike_rho_theta():
    with pytest.raises(ValueError):
        prtools.zernike(mask=1, index=1, normalize=True, rho=1, theta=None)


def test_zernike_basis():
    basis = prtools.zernike_basis(mask=np.ones((3, 3)), modes=1, vectorize=False)
    assert np.array_equal(basis, np.ones((1, 3, 3)))


def test_zernike_basis_vectorize():
    basis = prtools.zernike_basis(mask=np.ones((3, 3)), modes=1, vectorize=True)
    assert np.array_equal(basis, np.ones((1, 9)))


def test_zernike_fit():
    mask = prtools.circle((256, 256), 128, antialias=False)
    coeffs = np.random.default_rng(0).random(4)*100e-9
    opd = prtools.zernike_compose(mask, coeffs)
    fit_coeffs = prtools.zernike_fit(opd, mask, np.arange(1, 5))
    assert np.allclose(coeffs, fit_coeffs)


def test_zernike_remove():
    mask = prtools.circle((256, 256), 128, antialias=False)
    coeffs = np.random.default_rng(0).random(4)*100e-9
    opd = prtools.zernike_compose(mask, coeffs)
    residual = prtools.zernike_remove(opd, mask, np.arange(1, 5))
    assert np.allclose(residual, np.zeros_like(residual))


def test_zernike_random_mask():
    # nonzero mask values are treated as inside the mask
    mask = prtools.circle((256, 256), 128, antialias=False)
    rng = np.random.default_rng(0)
    random_mask = rng.uniform(low=-1, high=1, size=mask.shape) * mask
    assert np.array_equal(prtools.zernike(mask, 1), prtools.zernike(random_mask, 1))


def test_zernike_piston_is_float():
    mask = prtools.circle((64, 64), 30, antialias=False)
    assert prtools.zernike(mask, 1).dtype == np.float64


# Sample points on a 65x65 circular mask centered at (32, 32), as (row, col):
# +x edge, +y edge, and the 45 deg diagonal between them. +x is increasing
# column and +y is increasing row, so +y is the top of the image when
# displayed with origin='lower'
PLUS_X, PLUS_Y, PLUS_XY = (32, 60), (60, 32), (52, 52)


@pytest.mark.parametrize('index, positive_at', [
    (2, PLUS_X),     # x tilt
    (3, PLUS_Y),     # y tilt
    (5, PLUS_XY),    # 45 deg astigmatism
    (7, PLUS_Y),     # y coma
    (8, PLUS_X),     # x coma
])
def test_zernike_orientation(index, positive_at):
    # theta is measured from +x (increasing column) toward +y
    # (increasing row)
    mask = prtools.circle((65, 65), 32, antialias=False)
    z = prtools.zernike(mask, index)
    assert z[positive_at] > 0.5


def _zernike_reference(mask, j, origin, ysign=1):
    # Noll-normalized Zernike evaluated directly in the xy frame (x increasing
    # with column, y increasing with row) about the (row, col) origin. ysign=-1
    # mirrors the frame top-to-bottom, as in the legacy convention
    n, m = noll_to_nm(j)
    rr, cc = np.mgrid[0:mask.shape[0], 0:mask.shape[1]].astype(float)
    x, y = cc - origin[1], ysign * (rr - origin[0])
    r = np.hypot(x, y)
    rho = r / np.max(r * mask)
    theta = np.arctan2(y, x)
    if m == 0:
        Z = np.sqrt(n + 1) * R(m, n, rho) if n else np.ones_like(rho)
    elif m > 0:
        Z = np.sqrt(2*(n + 1)) * R(m, n, rho) * np.cos(m*theta)
    else:
        Z = np.sqrt(2*(n + 1)) * R(m, n, rho) * np.sin(-m*theta)
    return Z * mask


@pytest.mark.parametrize('n', [256, 255])
def test_zernike_matches_reference(n):
    # odd sizes check that the origin sits exactly on the mask centroid
    mask = prtools.circle((n, n), n//2 - 4, antialias=False)
    origin = prtools.centroid(mask)
    for j in range(1, 37):
        assert np.allclose(prtools.zernike(mask, j),
                           _zernike_reference(mask, j, origin), atol=1e-9)


def _same_coordinates(a, b):
    # compare (rho, theta) pairs as points in the plane. theta is +/-pi on the
    # -x axis and undefined at the origin, so a 1-ulp difference in the shift
    # can flip those pixels without changing the coordinates
    (rho_a, theta_a), (rho_b, theta_b) = a, b
    return np.allclose(rho_a*np.exp(1j*theta_a), rho_b*np.exp(1j*theta_b))


def test_zernike_coordinates_origin_at_mask_centroid():
    mask = prtools.circle((128, 128), 30, shift=(-20, 30), antialias=False,
                          indexing='ij')

    rho_xy, theta_xy = prtools.zernike_coordinates(mask, indexing='xy')
    rho_ij, theta_ij = prtools.zernike_coordinates(mask, indexing='ij')

    assert np.unravel_index(np.argmin(rho_xy), rho_xy.shape) == (44, 94)
    assert _same_coordinates((rho_xy, theta_xy), (rho_ij, theta_ij))


def test_zernike_coordinates_shift_indexing():
    mask = prtools.circle((128, 128), 30, shift=(-20, 30), antialias=False,
                          indexing='ij')

    auto = prtools.zernike_coordinates(mask)
    xy = prtools.zernike_coordinates(mask, shift=(30, -20), indexing='xy')
    ij = prtools.zernike_coordinates(mask, shift=(-20, 30), indexing='ij')

    assert _same_coordinates(xy, auto)
    assert _same_coordinates(ij, auto)


def test_zernike_coordinates_angle():
    # rotating the coordinates 90 deg from +x toward +y turns Z2 into Z3
    mask = prtools.circle((256, 256), 120, antialias=False)
    rho, theta = prtools.zernike_coordinates(mask, angle=90)
    assert np.allclose(prtools.zernike(mask, 2, rho=rho, theta=theta),
                       prtools.zernike(mask, 3))


def test_zernike_coeff_legacy_noll_signs():
    # sin terms (m < 0) flip; cos terms and m = 0 are unchanged
    expected = [1, 1, -1, 1, -1, 1, -1, 1, -1, 1, 1]
    assert np.array_equal(prtools.zernike_coeff_legacy(np.ones(11)), expected)


def test_zernike_coeff_legacy_involution():
    coeffs = np.random.default_rng(0).normal(size=15)
    assert np.array_equal(
        prtools.zernike_coeff_legacy(prtools.zernike_coeff_legacy(coeffs)), coeffs)


@pytest.mark.parametrize('order, modes', [
    ('noll', range(1, 16)),
    ('fringe', range(1, 17)),
    ('ansi', range(0, 15)),
])
def test_zernike_coeff_legacy_matches_mirror(order, modes):
    # legacy convention == current convention mirrored top-to-bottom (y -> -y)
    mask = prtools.circle((128, 128), 60, antialias=False)
    rho, theta = prtools.zernike_coordinates(mask)
    for j in modes:
        mirrored = prtools.zernike(mask, j, order=order, rho=rho, theta=-theta)
        sign = prtools.zernike_coeff_legacy([1.0], modes=[j], order=order)[0]
        assert np.allclose(mirrored, sign * prtools.zernike(mask, j, order=order))


@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize('order, modes', [
    ('noll', range(1, 16)),
    ('fringe', range(1, 17)),
    ('ansi', range(0, 15)),
])
def test_zernike_coeff_legacy_matches_legacy_flag(order, modes):
    # exact for even-sized binary masks; odd sizes also had a half-pixel
    # origin offset
    mask = prtools.circle((128, 128), 60, antialias=False)
    for j in modes:
        legacy = prtools.zernike(mask, j, order=order, legacy=True)
        sign = prtools.zernike_coeff_legacy([1.0], modes=[j], order=order)[0]
        assert np.allclose(legacy, sign * prtools.zernike(mask, j, order=order))


@pytest.mark.filterwarnings('ignore::FutureWarning')
@pytest.mark.parametrize('n', [256, 255])
def test_zernike_legacy_matches_reference(n):
    # legacy frame: y mirrored, origin measured from shape/2 instead of the
    # mesh origin at floor(shape/2), so odd sizes sit half a pixel off
    mask = prtools.circle((n, n), n//2 - 4, antialias=False)
    centroid = np.asarray(prtools.centroid(mask))
    origin = np.floor(np.asarray(mask.shape)/2) + centroid - np.asarray(mask.shape)/2
    for j in range(1, 37):
        assert np.allclose(prtools.zernike(mask, j, legacy=True),
                           _zernike_reference(mask, j, origin, ysign=-1),
                           atol=1e-9)


@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_zernike_legacy_binary_mask():
    # legacy only changes orientation; masks are still treated as binary
    mask = prtools.circle((64, 64), 30, antialias=False)
    rng = np.random.default_rng(0)
    random_mask = rng.uniform(low=0.1, high=1, size=mask.shape) * mask
    for j in range(1, 12):
        assert np.array_equal(prtools.zernike(mask, j, legacy=True),
                              prtools.zernike(random_mask, j, legacy=True))


LEGACY_CALLS = {
    'zernike':         lambda m, **kw: prtools.zernike(m, 4, **kw),
    'zernike_compose': lambda m, **kw: prtools.zernike_compose(m, [0, 1e-8], **kw),
    'zernike_basis':   lambda m, **kw: prtools.zernike_basis(m, [2, 3], **kw),
    'zernike_fit':     lambda m, **kw: prtools.zernike_fit(np.zeros(m.shape), m, [2, 3], **kw),
    'zernike_remove':  lambda m, **kw: prtools.zernike_remove(np.zeros(m.shape), m, [2, 3], **kw),
}


@pytest.mark.parametrize('name', LEGACY_CALLS)
def test_zernike_legacy_warns(name):
    mask = prtools.circle((32, 32), 14, antialias=False)
    with pytest.warns(FutureWarning, match='prtools 2.0.2'):
        LEGACY_CALLS[name](mask, legacy=True)
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        LEGACY_CALLS[name](mask)


def test_zernike_coeff_legacy_stacked():
    out = prtools.zernike_coeff_legacy(np.ones((5, 11)))
    assert out.shape == (5, 11)
    assert np.array_equal(out[0], out[-1])


def test_zernike_coeff_legacy_modes_length_mismatch():
    with pytest.raises(ValueError):
        prtools.zernike_coeff_legacy([1, 2], modes=[1])


def test_zernike_coeff_legacy_bad_input():
    with pytest.raises(ValueError):
        prtools.zernike_coeff_legacy(1.0)
    with pytest.raises(ValueError):
        prtools.zernike_coeff_legacy([1.0], order='zemax')
