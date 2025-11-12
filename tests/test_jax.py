import numpy as np
import jax.numpy as jnp

import prtools

def test_lbfgs_fn_kwargs():
    prtools.use('jax')

    def fn(x, a, b):
        return (x[0] + a)**2 + (x[1] + b)**2
    
    a = 2
    b = -5
    x0 = jnp.zeros(2)

    res = prtools.jax.lbfgs(fn, x0, gtol=1e-5, maxiter=25, 
                            fn_kwargs={'a': a, 'b': b})
    assert(np.allclose(res.x, (-a, -b)))