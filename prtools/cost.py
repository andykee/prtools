from prtools._array_api import array_namespace


def sserror(data, est, mask=None, gain_bias_invariant=False, ghat=None):
    r"""Compute the normalized sum squared error between two arrays.

    The error between measured data :math:`\tilde{s}` and estimated data
    :math:`\hat{s}` is computed as the gain invariant error metric

    .. math::

        \Phi = \frac{1}{K}\sum_k
        \frac{\sum_{x,y} w(x,y,k)\left[\alpha_k\hat{s}(x,y,k) -
        \tilde{s}(x,y,k)\right]^2}{\sum_{x,y} w(x,y,k)\tilde{s}^2(x,y,k)}

    for :math:`k = 1, 2, ... K` independent measurements, where
    :math:`w` is a weighting function given by `mask` and each
    :math:`\alpha_k` is chosen to minimize the error. The result is
    independent of any relative scaling (gain) difference between the inputs.

    If `gain_bias_invariant` is ``True``, the gain and bias invariant error
    metric is computed instead

    .. math::

        \Phi = \frac{1}{K}\sum_k
        \frac{\sum_{x,y} w(x,y,k)\left[\alpha_k\hat{g}(x,y,k) -
        \tilde{g}(x,y,k)\right]^2}{\sum_{x,y} w(x,y,k)\tilde{s}^2(x,y,k)}

    where :math:`\hat{g}` and :math:`\tilde{g}` are :math:`\hat{s}` and
    :math:`\tilde{s}` with their weighted means subtracted and each
    :math:`\alpha_k` is chosen to minimize the error. The result is
    independent of any relative scaling (gain) and offset (bias) differences
    between the inputs.

    Parameters
    ----------
    data : array_like
        Ideal or measured data with shape (M, N) or (K, M, N)
    est : array_like
        Estimated or modeled data with the same shape as `data`
    mask : array_like, optional
        Weighting function :math:`w` applied to the inputs. Entries equal to
        1 (or True) are fully included in the error computation and entries
        equal to 0 (or False) are ignored. Intermediate values partially
        weight the corresponding elements. Mask must either have the same
        shape as the inputs or be broadcastable to the same shape. If None
        (default), all elements are equally weighted.
    gain_bias_invariant : bool, optional
        If True, the error is computed to be independent of any gain or bias
        differences between the inputs. If False (default), the error is
        computed to be independent of any gain differences between the inputs.
    ghat : array_like, optional
        Precomputed bias-subtracted estimate :math:`\hat{g}`. Only used if
        `gain_bias_invariant` is True. If None (default), :math:`\hat{g}`
        is computed from `est`.

    Returns
    -------
    error : float
        Normalized sum squared error

    References
    ----------
    [1] S. Thurman and J. Fienup, "Phase retrieval with signal bias", J. Opt. Soc. Am. A/Vol. 26, No. 4 (2009)

    [2] A. Jurling and J. Fienup, "Applications of algorithmic differentiation to phase retrieval algorithms", J. Opt. Soc. Am. A/Vol. 31, No. 7 (2014)

    """
    xp = array_namespace(data, est)

    if gain_bias_invariant:
        if ghat is None:
            ghat = g(est, mask, xp)  # Eq. 12
        mtil = g(data, mask, xp)  # Eq. 13
        mhat = ghat
    else:
        mtil = data
        mhat = est

    alpha = _a(mtil, mhat, mask, xp)  # Eq. 4 or Eq. 14
    alpha = alpha[..., xp.newaxis, xp.newaxis]

    num = _wsum(xp.square(alpha * mhat - mtil), mask, xp)
    den = _wsum(xp.square(data), mask, xp)

    # Eq. 3 or Eq. 11
    return xp.mean(num/den)


def _wsum(x, mask, xp):
    # Weighted sum over the last two axes
    if mask is not None:
        x = mask * x
    return xp.sum(x, axis=(-2, -1))


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
    # Eq. 4 or Eq. 14 in [1]
    try:
        return _wsum(gtil * ghat, mask, xp)/_wsum(xp.square(ghat), mask, xp)
    except FloatingPointError as e:
        if xp.all(ghat == 0):
            raise ZeroDivisionError('ghat must be nonzero')
        else:
            raise e
