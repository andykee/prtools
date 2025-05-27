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

from prtools import __backend__


# TODO: The interface for registering a dataclass with JAX was significantly
# improved in JAX v0.4.36. Once some time has passed, we should switch over
# to this new method (and require JAX >= 0.4.36)
@partial(jax.tree_util.register_dataclass,
         data_fields=['x', 'n', 'grad', 'fun'],
         meta_fields=[])
@dataclass
class JaxOptimizeResult:
    """Represents the optimization result."""
    x: jax.Array  #: The solution of the optimization
    n: jax.Array  #: Number of iterations performed by the optimizer
    grad: jax.Array  #: Value of objective function gradient at x
    fun: jax.Array  #: Value of objective function at x


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
