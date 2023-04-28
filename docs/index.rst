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
   medfix2
   normpow
   shift
   register

Statistical functions
---------------------
.. autosummary::
   :toctree: generated
   :caption: Statistical functions
   :nosignatures:
   

   encircled_energy
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
   gauss2
   sin2
   waffle2

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

   min_sampling
   pixelscale_nyquist
