from dataclasses import dataclass
from typing import Any

from prtools import __backend__
from prtools.backend import JAX_AVAILABLE

if JAX_AVAILABLE:
    import jax
    import optax
    import optax.tree_utils as otu


def register_dataclass(cls):
    if JAX_AVAILABLE:
        data_fields = ['x', 'n', 'grad', 'value', 'state']
        meta_fields = []
        cls = jax.tree_util.register_dataclass(cls,
                                               data_fields=data_fields,
                                               meta_fields=meta_fields)
    return cls


@register_dataclass
@dataclass
class JaxOptimizeResult:
    """Represents the optimization result."""
    x: Any  #: The solution of the optimization
    n: Any  #: Number of iterations performed by the optimizer
    value: Any  #: Value of objective function at x
    grad: Any  #: Value of objective function gradient at x
    state: Any  #: Optimizer state


def lbfgs(fn, x0, gtol=None, maxiter=None, callback=None, fn_args=None,
          fn_kwargs=None):
    """Minimize a scalar function of one or more variables using the L-BFGS
    algorithm

    Parameters
    ----------
    fn : callable
        The objective function to be minimized:

        .. code:: python

            fn(x, *fn_args, **fn_kwargs)

        where ``x`` is a 1-D array with shape (n,) and ``fn_args`` and
        ``fn_kwargs`` are optional positional and keyword arguments.
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
    fn_args : iterable or None
        Extra positional arguments passed to ``fn()``
    fn_kwargs : dict or None
        Extra keyword arguments passed to the ``fn()``

    Returns
    -------
    res: JaxOptimizeResult
        The optimization result. See :class:`JaxOptimizeResult` for a
        description of attributes.

    """
    if not JAX_AVAILABLE:
        message = "jax and optax must be installed to use method `lbfgs`."
        raise ModuleNotFoundError(message)

    if __backend__ != 'jax':
        raise RuntimeError('JAX backend must be selected')

    if not any((gtol, maxiter)):
        raise ValueError('At least one termination tolerance must be specified.')

    fn_args = [] if fn_args is None else fn_args
    fn_kwargs = {} if fn_kwargs is None else fn_kwargs

    opt = optax.lbfgs()
    value_and_grad_fn = optax.value_and_grad_from_state(fn)

    def step(carry):
        params, state = carry
        # NOTE: passing *args and **kwargs to value_and_grad_fun is very
        # poorly documented in optax (as of v0.2.6 - 10/2025) but this
        # seems to work for now
        value, grad = value_and_grad_fn(params, *fn_args, state=state,
                                        **fn_kwargs)
        updates, state = opt.update(
            grad, state, params, value=value, grad=grad, value_fn=fn)
        if callback:
            res = JaxOptimizeResult(
                n=otu.tree_get(state, 'count')-1,
                x=params,
                grad=grad,
                value=value,
                state=state)
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
        continuing_criterion, step, init_carry)

    res = JaxOptimizeResult(
        n=otu.tree_get(final_state, 'count'),
        x=final_params,
        grad=otu.tree_get(final_state, 'grad'),
        value=otu.tree_get(final_state, 'value'),
        state=final_state)

    if callback:
        jax.debug.callback(callback, res)

    return res


def _multi_dot_three(a, b, c, axes, out):
    # compute the matrix triple product
    #
    # a few notes:
    # * while numpy-based implementation of this method is based on
    #   np.matmul, jax.numpy.matmul doesn't implement the axes argument so
    #   we have to use jax.numpy.linalg.multi_dot instead
    # * the implementation used here supports b with ndim in (2, 3)
    #   iterating over any of the 3 axes when b.ndim == 3
    # * jax.vmap handles the case when b.ndim == 3 compared with the numpy
    #   equivalent of this function which does everything within the
    #   confines of matmul using the axes argument
    if b.ndim == 2:
        return jax.numpy.linalg.multi_dot((a, b, c))
    else:
        iter_axis = _iter_axis(axes)
        return jax.vmap(_multi_dot, in_axes=[None, iter_axis, None], out_axes=iter_axis)(a, b, c)

def _multi_dot(a, b, c):
        # wrapper function to support vmap call signature
        return jax.numpy.linalg.multi_dot((a, b, c))


def _iter_axis(axes):
    # pure Python to avoid dealing with JAX array mutability issues
    mask = [0, 1, 2]
    for ax in axes:
        mask[ax] = None
    return [ax for ax in mask if ax is not None][0]


def _ndrebin(a, f):
    return jax.vmap(_rebin, in_axes=[0, None])(a, f)


def _rebin(a, f):
        return jax.numpy.reshape(a, (a.shape[0]//f, f, a.shape[1]//f, f)).sum(-1).sum(1)