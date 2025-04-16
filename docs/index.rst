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

Backends
--------
:mod:`prtools` supports multiple numerical backends for representing
N-dimensional data and performing numerical calculations. Currently Numpy (the
default backend) and JAX are supported. The :mod:`prtools` interface remains
the same, but a different backend library is used under the hood.

.. autosummary::
   :toctree: generated
   :caption: Backends
   :nosignatures:

   set_backend
   get_backend

.. important::

   Numpy is a :mod:`prtools` dependency and will be automatically installed as
   needed. No additional backends are installed (or required to use
   :mod:`prtools`). To install additional backends, refer to their respective
   installation instructions:

   * `JAX <https://docs.jax.dev/en/latest/developer.html#building-or-installing-jaxlib>`_
     
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

   spindex
   sparray
   spmatrix
   spindex_from_mask
   mask_from_spindex

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

Optimization
------------
.. autosummary::
   :toctree: generated
   :caption: Optimization
   :nosignatures:

   lbfgs

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

