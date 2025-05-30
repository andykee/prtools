**prtools** --- Utility functions for image-based phase retrieval
=================================================================

**Version**: |version|

**Useful links**:
`Source Repository <https://github.com/andykee/prtools>`_ |
`Issue Tracker <https://github.com/andykee/prtools/issues>`_ |
`Releases <https://github.com/andykee/prtools/releases>`_

The :mod:`prtools` library provides functionality and tools that may be
useful when performing image-based phase retrieval including:

* Image processing, reduction, and analysis
* Creating, fitting, and removing Zernike polynomials
* Drawing various shapes in arrays
* Computing statistics and optical simulation parameters

Install :mod:`prtools` with pip:

.. code-block:: bash

    pip install prtools

.. currentmodule:: prtools

:mod:`prtools` supports different numerical backends for representing
N-dimensional data and performing numerical calculations. The backend is
configured by the :func:`prtools.use` function. Currently 
`Numpy <https://numpy.org>`_ (the default backend) and 
`JAX <https://docs.jax.dev>`_ are supported. The table below shows how to use
and optionally install each of the available backends:

========== ============================ ========================
Backend    Install                      Use
========== ============================ ========================
numpy [1]_ ``pip install prtools``      ``prtools.use('numpy')``
jax        ``pip install prtools[jax]`` ``prtools.use('jax')``  
========== ============================ ========================

The current backend name is given in the ``prtools.__backend__`` attribute.

.. note::

   The :mod:`prtools` API remains the same regardless of which backend is in
   use, but different backends may expose additional functionality. See the
   :ref:`backend-specific documentation <backends>` below for more details.

.. [1] The Numpy backend is initialized by default when :mod:`prtools` is
   imported. No additional steps are required to use :mod:`prtools` with the
   Numpy backend.

Array manipulation
------------------
.. autosummary::
   :toctree: generated
   :caption: Array manipulation
   :nosignatures:
   
   centroid
   pad
   subarray
   boundary
   rebin
   rescale
   medfix
   normpow
   shift
   register

Array metrics
-------------
.. autosummary::
   :toctree: generated
   :caption: Array metrics
   :nosignatures:
   
   rms
   pv
   radial_avg
   ee

Shapes
------
.. autosummary::
   :toctree: generated
   :caption: Shapes
   :nosignatures:

   circle
   rectangle
   hexagon
   hex_segments
   spider
   gauss
   sin
   waffle
   mesh

.. note::

   The shape functions support both Cartesian (``xy``) and matrix (``ij``)
   indexing conventions for specifying the shift parameter via the 
   ``indexing`` parameter. The default is matrix (``indexing='ij'``) for 
   all functions.

Fourier transforms
------------------
.. autosummary::
   :toctree: generated
   :caption: Fourier transforms
   :nosignatures:

   dft2
   idft2

Sparse matrices
---------------
.. autosummary::
   :toctree: generated
   :caption: Sparse matrices
   :nosignatures:

   index
   sparse
   dense
   index_from_mask
   mask_from_index

Zernike polynomials
-------------------
.. autosummary::
   :toctree: generated
   :caption: Zernike polynomials
   :nosignatures:
   
   zernike
   zernike_fit
   zernike_remove
   zernike_compose
   zernike_basis
   zernike_coordinates

Cost functions
--------------
.. autosummary::
   :toctree: generated
   :caption: Cost functions
   :nosignatures:

   sserror

Miscellaneous
-------------
.. autosummary::
   :toctree: generated
   :caption: Miscellaneous
   :nosignatures:

   calcpsf
   pixelscale_nyquist
   min_sampling
   fft_shape
   translation_defocus
   find_wrapped

Numerical backends
------------------
.. autosummary::
   :toctree: generated
   :caption: Numerical backends
   :nosignatures:

   use

.. _backends:

JAX backend
-----------
The JAX backend has the following additional dependencies:

* `JAX <https://docs.jax.dev/>`__
* `Optax <https://optax.readthedocs.io/en/latest/>`_

.. autosummary::
   :toctree: generated
   :caption: JAX backend
   :nosignatures:

   jax.lbfgs
   jax.JaxOptimizeResult
