import numpy as np
import pytest

import prtools


def test_mesh_xy_matches_meshgrid():
    # x increases with column and y increases with row, about the center
    # pixel at (nrows//2, ncols//2)
    x, y = prtools.mesh((4, 5), indexing='xy')
    xx, yy = np.meshgrid(np.arange(5) - 2, np.arange(4) - 2, indexing='xy')
    assert np.array_equal(x, xx)
    assert np.array_equal(y, yy)


def test_mesh_xy_shift():
    x0, y0 = prtools.mesh((5, 6), indexing='xy')
    x, y = prtools.mesh((5, 6), shift=(1, 2), indexing='xy')
    assert np.array_equal(x, x0 - 1)
    assert np.array_equal(y, y0 - 2)


@pytest.mark.parametrize('shape', [(64, 64), (63, 65)])
@pytest.mark.parametrize('angle', [0, 30, -110])
def test_mesh_xy_is_ij_axis_swap(shape, angle):
    # xy is ij with the axes swapped: shift order swaps and the rotation
    # changes sign
    r, c = prtools.mesh(shape, shift=(3.5, -2), angle=angle, indexing='ij')
    x, y = prtools.mesh(shape, shift=(-2, 3.5), angle=-angle, indexing='xy')
    assert np.allclose(x, c)
    assert np.allclose(y, r)


def test_mesh_xy_rotation():
    # an xy rotation goes from +x toward +y: after rotating 90 deg, the point
    # 5 pixels along +y lies on the rotated +x axis
    x, y = prtools.mesh((33, 33), angle=90, indexing='xy')
    assert np.allclose((x[16 + 5, 16], y[16 + 5, 16]), (5, 0))


def test_mesh_invalid_indexing():
    with pytest.raises(ValueError):
        prtools.mesh((8, 8), indexing='rc')


SHAPES = {
    'circle':    lambda i, s, a: prtools.circle((128, 127), 30, shift=s, indexing=i),
    'hexagon':   lambda i, s, a: prtools.hexagon((128, 127), 30, shift=s, indexing=i),
    'rectangle': lambda i, s, a: prtools.rectangle((128, 127), 50, 20, shift=s, angle=a, indexing=i),
    'ellipse':   lambda i, s, a: prtools.ellipse((128, 127), 40, 20, shift=s, angle=a, indexing=i),
    'spider':    lambda i, s, a: prtools.spider((128, 127), 3, angle=a, shift=s, indexing=i),
    'sin':       lambda i, s, a: prtools.sin((128, 127), 4, shift=s, angle=a, indexing=i),
    'waffle':    lambda i, s, a: prtools.waffle((128, 127), 4, shift=s, angle=a, indexing=i),
}


@pytest.mark.parametrize('name', SHAPES)
def test_shape_xy_is_ij_axis_swap(name):
    # indexing='xy' takes (x, y) shifts and rotates from +x toward +y, so it
    # must draw the same array as 'ij' with the shift swapped and the angle
    # negated
    f = SHAPES[name]
    assert np.allclose(f('xy', (13.2, -9.7), 33), f('ij', (-9.7, 13.2), -33))


def test_circle_shift():
    circ = prtools.circle((128, 128), 10, shift=(20, 15), indexing='xy')
    assert np.allclose(prtools.centroid(circ, mode='center', indexing='xy'), (20, 15))
    assert np.allclose(prtools.centroid(circ, mode='center', indexing='ij'), (15, 20))


@pytest.mark.parametrize('indexing', ['ij', 'xy'])
def test_rectangle_width_is_horizontal(indexing):
    rect = prtools.rectangle((101, 101), 60, 20, antialias=False, indexing=indexing)
    assert np.any(rect, axis=0).sum() == 61
    assert np.any(rect, axis=1).sum() == 21


@pytest.mark.parametrize('rotate', [False, True])
def test_hexagon_orientation_independent_of_indexing(rotate):
    ij = prtools.hexagon((201, 201), 80, rotate=rotate, antialias=False, indexing='ij')
    xy = prtools.hexagon((201, 201), 80, rotate=rotate, antialias=False, indexing='xy')
    assert np.array_equal(ij, xy)


@pytest.mark.parametrize('indexing, sign', [('ij', -1), ('xy', 1)])
def test_rectangle_angle(indexing, sign):
    # ij rotates toward decreasing row (counterclockwise with origin='upper'),
    # xy rotates toward increasing row (counterclockwise with origin='lower')
    rect = prtools.rectangle((201, 201), 160, 4, angle=30, antialias=False,
                             indexing=indexing)
    rr, cc = np.nonzero(rect)
    slope = np.polyfit(cc - 100, rr - 100, 1)[0]
    assert np.isclose(slope, sign * np.tan(np.radians(30)), atol=0.02)


@pytest.mark.parametrize('indexing, shift, axis', [
    ('ij', (0, 10), 1),   # +col
    ('ij', (10, 0), 0),   # +row
    ('xy', (10, 0), 1),   # +x is +col
    ('xy', (0, 10), 0),   # +y is +row
])
def test_sin_shift(indexing, shift, axis):
    # a shift translates the pattern by exactly that many pixels
    angle = 0 if axis == 1 else (90 if indexing == 'xy' else -90)
    s0 = prtools.sin((300, 300), 4, angle=angle, indexing=indexing)
    s1 = prtools.sin((300, 300), 4, shift=shift, angle=angle, indexing=indexing)
    assert np.allclose(np.take(s1, range(10, 300), axis=axis),
                       np.take(s0, range(0, 290), axis=axis))
