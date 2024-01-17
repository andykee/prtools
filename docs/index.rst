:mod:`prtools` --- Utility functions for image-based phase retrieval
====================================================================

The :mod:`prtools` package implements functionality and tools that may be
helpful when performing image-based phase retrieval including:

* Image processing, reduction, and analysis
* Creating, fitting, and removing Zernike polynomials
* Drawing various shapes in arrays
* Computing statistics and optical simulation parameters

.. currentmodule:: prtools

Array manipulation
------------------
.. autosummary::
   :toctree: generated
   :caption: Array manipulation
   :nosignatures:
   
   centroid
   pad
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
   
   ee
   rms
   pv

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

Shapes
------
.. autosummary::
   :toctree: generated
   :caption: Shapes
   :nosignatures:

   circle
   circlemask
   rectangle
   hexagon
   gauss
   sin
   waffle

.. note::

   The shape functions support both Cartesian (``xy``) and matrix (``ij``)
   indexing conventions for specifying the shift parameter via the 
   ``indexing`` parameter. The default is Cartesian (``indexing='xy'``) for 
   all functions.

Misc utilities
--------------
.. autosummary::
   :toctree: generated
   :caption: Misc utilities
   :nosignatures:

   calcpsf
   pixelscale_nyquist
   min_sampling
   fft_shape
   translation_defocus

Sparse matrices
---------------
.. autosummary::
   :toctree: generated
   :caption: Sparse matrices
   :nosignatures:

   sparray
   spmatrix
   spindex
   spmask
   
Fourier transforms
------------------
.. autosummary::
   :toctree: generated
   :caption: Fourier transforms
   :nosignatures:

   dft2
   idft2
