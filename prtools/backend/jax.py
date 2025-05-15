from dataclasses import dataclass
from functools import partial
import importlib

try:
    import jax
    import optax
    import optax.tree_utils as otu
except ImportError as exc:
    JAX_AVAILABLE = False
else:
    JAX_AVAILABLE = True

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


# TODO: The interface for registering a dataclass with JAX was significantly
# improved in JAX v0.4.36. Once some time has passed, we should switch over
# to this new method (and require JAX >= 0.4.36)
@partial(jax.tree_util.register_dataclass,
         data_fields=['x', 'n', 'grad', 'fun'],
         meta_fields=[])
@dataclass
class JaxOptimizeResult:
    """Represents the optimization result.

    Attributes
    ----------
    x : jax.Array
        The solution of the optimization.
    fun : jax.Array
        Value of objective function at x.
    grad : jax.Array
        Values of objective function's gradient at x.
    n : jax.Array
        Number of iterations performed by the optimizer.
    """
    x: jax.Array  # The solution of the optimization
    n: jax.Array
    grad: jax.Array
    fun: jax.Array


def lbfgs(fun, x0, gtol, maxiter, callback=None):
    """Minimize a scalar function of one or more variables using the L-BFGS
    algorithm

    Parameters
    ----------
    fun : callable
        The objective function to be minimied
    x0 : jax.Array
        Initial guess
    gtol : float
        Iteration stops when ``l2_norm(grad) <= gtol``
    maxiter : int
        Maximum number of iterations
    callback : callable, optional
        A callable called after each iteration with the signature

        .. code:: python

            callback(intermediate_result: JaxOptimizeResult)

        where ``intermediate_result`` is a :class:`JaxOptimizeResult`.

    Returns
    -------
    res: JaxOptimizeResult
        The optimization result. See :class:`JaxOptimizeResult` for a
        description of attributes.

    """
    if not JAX_AVAILABLE:
        message = "jax and optax must be installed to use method `lbfgs`."
        raise ModuleNotFoundError(message) from exc

    if __backend__ != 'jax':
        raise RuntimeError('JAX backend must be selected')

    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun,
        )
        if callback:
            res = JaxOptimizeResult(
                n=otu.tree_get(state, 'count'),
                x=params, 
                grad=grad,
                fun=value)
            jax.debug.callback(callback, res)
        params = optax.apply_updates(params, updates)
        return params, state

    def continuing_criterion(carry):
        _, state = carry
        iter_num = otu.tree_get(state, 'count')
        grad = otu.tree_get(state, 'grad')
        grad_norm = otu.tree_l2_norm(grad)
        return (iter_num == 0) | ((iter_num < maxiter) & (grad_norm >= gtol))

    init_carry = (x0, opt.init(x0))
    final_params, final_state = jax.lax.while_loop(
        continuing_criterion, step, init_carry
    )
    return final_params, final_state
