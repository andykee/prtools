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


Array API support
-----------------
:mod:`prtools` supports the `Python array API standard 
<https://data-apis.org/array-api/latest/index.html>`_. This means that 
:mod:`prtools` functions should work seamlessly with any array API-compatible
ararys like those provided by NumPy, CuPy, PyTorch, and JAX. 

.. note::

   Only NumPy and JAX interoperability has been tested to date. Future backend
   libraries may be tested in the future.


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
   nanrms
   pv
   nanpv
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

Convolution and filtering
-------------------------
.. autosummary::
   :toctree: generated
   :caption: Convolution and filtering
   :nosignatures:

   fftconv
   gauss_blur
   pixelate
   gauss
   sinc
   gauss_kernel
   pixel_kernel

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

``jax`` module
--------------
The ``jax`` module has the following additional dependencies:

* `JAX <https://docs.jax.dev/>`_
* `Optax <https://optax.readthedocs.io/en/latest/>`_

.. autosummary::
   :toctree: generated
   :caption: jax module
   :nosignatures:

   jax.lbfgs
   jax.JaxOptimizeResult
