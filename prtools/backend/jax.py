import importlib
from typing import NamedTuple

from ._base import _BackendLibrary
from prtools import __backend__
from prtools._backend import numpy as np


class Numpy(_BackendLibrary):
    def __init__(self):
        super().__init__(importlib.import_module('jax.numpy'))

    def broadcast_to(self, array, shape):
        # jax broadcast_to expects an array input
        array = self.module.asarray(array)
        return self.module.broadcast_to(array, shape)

    def dot(self, a, b, out=None):
        # jax.numpy.dot doesn't support the `out` parameter so we ignore it
        return self.module.dot(a, b)

    def max(self, a, *args, **kwargs):
        # jax max expects an array input
        array = self.module.asarray(a)
        return self.module.max(array, *args, **kwargs)

    def multiply(self, a, b, out=None):
        # jax.numpy.multiply doesn't support the `out` parameter so we
        # ignore it
        return self.module.multiply(a, b)

    def divide(self, a, b, out=None):
        # jax.numpy.divide doesn't support the `out` parameter so we
        # ignore it
        return self.module.divide(a, b)


class Scipy(_BackendLibrary):
    def __init__(self):
        super().__init__(importlib.import_module('jax.scipy'))


class OptimizeResult(dict):
    """Represents the optimization result.

    Attributes
    ----------
    x : ndarray
        The solution of the optimization.
    
    """

    def __init__(self):
        super().__init__()

    #def __getattr__(self, name):
    #    try:
    #        return self[name]
    #    except KeyError as e:
    #        raise AttributeError(name) from e
        
    #__setattr__ = dict.__setitem__
    #__delattr__ = dict.__delitem__


def lbfgs(fun, x0, tol, maxiter, callback=None):
    """Minimize a scalar function of one or more variables using the L-BFGS
    algorithm

    Parameters
    ----------
    fun : callable
        The objective function to be minimied
    x0 : array_like
        Initial starting guess
    tol : float
        Termination tolerance
    maxiter : int
        Maximum number of iterations
    callback : callable, optional
        Not currently implemented

    Returns
    -------
    final_params :

    final_state :

    """
    if __backend__ != 'jax':
        raise RuntimeError('JAX backend must be selected')

    try:
        import jax
        import optax
        import optax.tree_utils as otu
    except ImportError as error:
        raise ImportError('lbfgs requires jax and optax') from error

    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun,
        )
        if callback:
            res = {}
            res['x'] = params
            res['grad'] = grad
            res['fun'] = value
            res['nit'] = otu.tree_get(state, 'count')
            jax.debug.callback(callback, res)
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        err = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < maxiter) & (err >= tol))

    init_carry = (x0, opt.init(x0))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state
