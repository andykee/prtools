from prtools._array_api import array_namespace


def sserror(data, est, mask=None, gain_bias_invariant=False, ghat=None):
    r"""Compute the normalized sum squared error between two arrays.

    The normalized sum squared error between ideal data `f` and estimated data
    `g` is given by

    .. math::

        \mbox{error} = \frac{1}{N}\sum_n \frac{\sum{\left|g(n,x,y)-f(n,x,y)\right|^2}}{\sum{\left|f(n,x,y)\right|^2}}

    for :math:`n = 1, 2, ... N` independent measurements.

    If `gain_bias_invariant` is ``True``, the error metric is computed such
    that the result is independent of relative scaling (gain) and offset
    (bias) differences between `f` and  `g` [1].

    Parameters
    ----------
    data : array_like
        Ideal or measured data
    est : array_like
        Estimated or modeled data
    mask : array_like, optional
        Mask applied to the inputs where a True value indicates that the
        corresponding element of the array is invalid. Mask must either have
        the same shape as the inputs or be broadcastable to the same shape and
        contain entries that are castable to bool. If None (default), the
        inputs are not masked.
    gain_bias_invariant : bool, optional
        If True, the error is computed to be independent of any
        gain or bias differences between the inputs. Default is False.

    Returns
    -------
    error : float
        Normalized root mean squared error

    References
    ----------
    [1] S. Thurman and J. Fienup, "Phase retrieval with signal bias", J. Opt. Soc. Am. A/Vol. 26, No. 4 (2009)

    [2] A. Jurling and J. Fienup, "Applications of algorithmic differentiation to phase retrieval algorithms", J. Opt. Soc. Am. A/Vol. 31, No. 7 (2014)

    """
    xp = array_namespace(data, est)
    shat = est  # estimated
    stil = data   # measured

    if shat.ndim == 3:
        K = shat.shape[0]
    else:
        K = 1

    if ghat is None:
        ghat = g(shat, mask, xp)
    gtil = g(stil, mask, xp)
    alpha = _a(gtil, ghat, mask, xp)

    if gain_bias_invariant:
        s = 'abcdefghijklmnopqrstuvwxyz'
        subs = f'{s[0:ghat.ndim]},{s[0:alpha.ndim]}->{s[0:ghat.ndim]}'

        if mask is None:
            resid = xp.einsum(subs, ghat, alpha) - gtil
            sse = 1/K * xp.sum(xp.square(resid))/xp.sum(xp.square(stil))
        else:
            resid = mask * (xp.einsum(subs, ghat, alpha) - gtil)
            sse = 1/K * xp.sum(mask * xp.square(resid))/xp.sum(mask * xp.square(stil))

    else:
        if mask is None:
            resid = ghat - gtil
            num = xp.sum(xp.square(ghat - gtil), axis=(-2, -1))
            den = xp.sum(xp.square(stil), axis=(-2, -1))
        else:
            num = xp.sum(mask * xp.square(ghat - gtil), axis=(-2, -1))
            den = xp.sum(mask * xp.square(stil), axis=(-2, -1))
        sse = 1/K * xp.sum(num/den)

    return sse


def g(data, mask, xp):
    # This function implements Eqs. 12 and 13 in [1]
    if mask is None:
        numer = xp.sum(data, axis=(-2, -1))
        denom = xp.prod(xp.asarray(data.shape[-2:]))
    else:
        numer = xp.sum(mask * data, axis=(-2, -1))
        denom = xp.sum(mask, axis=(-2, -1))
    x = numer/denom
    if data.ndim == 3:
        x = x[:, xp.newaxis, xp.newaxis]
    return data - x


def _a(gtil, ghat, mask, xp):
    # Eq. 14 in [1]
    try:
        if mask is None:
            num = gtil * ghat
            den = xp.square(ghat)
        else:
            num = mask * gtil * ghat
            den = mask * xp.square(ghat)
        return xp.sum(num, axis=(-2, -1))/xp.sum(den, axis=(-2, -1))

    except FloatingPointError as e:
        if xp.all(ghat) == 0:
            raise ZeroDivisionError('ghat must be nonzero')
        else:
            raise e
