from prtools.backend import numpy as np
from prtools import get_backend


def lbfgs(fun, x0, tol, maxiter, callback=None):
    """Minimize a scalar function of one or more variables using the L-BFGS
    algorithm

    .. attention::

        `optax <https://optax.readthedocs.io/en/latest/#installation>`_ must
        be installed before using :func:`lbfgs`.

        The JAX backend must be active to use this function.

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
    try:
        import optax
        import optax.tree_utils as otu
        import jax
    except ImportError:
        ImportError('optax must be installed to use lbfgs')

    if get_backend() != 'jax':
        raise RuntimeError('prtools must be using JAX backend to use lbfgs')

    if callback:
        # https://optax.readthedocs.io/en/latest/_collections/examples/lbfgs.html#l-bfgs-solver
        raise NotImplementedError

    opt = optax.lbfgs()
    value_and_grad_fun = optax.value_and_grad_from_state(fun)

    def step(carry):
        params, state = carry
        value, grad = value_and_grad_fun(params, state=state)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fun,
        )
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
