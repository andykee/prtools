:html_theme.sidebar_secondary.remove:

:mod:`prtools` --- Utility functions for image-based phase retrieval
====================================================================

The :mod:`prtools` package implements functionality and tools that may be
helpful when performing image-based phase retrieval including:

* Image processing, reduction, and analysis
* Creating, fitting, and removing Zernike polynomials
* Drawing various shapes in arrays
* Computing statistics and optical simulation parameters

.. currentmodule:: prtools

**Array manipulation:**

.. autosummary::
   :nosignatures:

   centroid
   pad
   boundary
   rebin
   rescale
   medfix2
   normalize_power
   shift
   register

**Statistical functions:**

.. autosummary::
   :nosignatures:

   encircled_energy
   rms
   pv
   strehl

**Zernike polynomials:**

.. autosummary::
   :nosignatures:

   zernike
   zernike_fit
   zernike_remove
   zernike_compose
   zernike_basis

**Shapes:**

.. note::

   The shape functions support both Cartesian (``xy``) and matrix (``ij``)
   indexing conventions for specifying the shift parameter via the 
   ``indexing`` parameter. The default is Cartesian (``indexing='xy'``) for 
   all functions.

.. autosummary::
   :nosignatures:

   circle
   circlemask
   rectangle
   hexagon
   gauss2
   sin2
   waffle2

**Misc utilities:**

.. autosummary::
   :nosignatures:

   min_sampling
   pixelscale_nyquist


.. raw:: html

   <h2>prtools functions</h2>

.. autofunction:: boundary
.. autofunction:: centroid
.. autofunction:: circle
.. autofunction:: circlemask
.. autofunction:: encircled_energy
.. autofunction:: gauss2
.. autofunction:: hexagon
.. autofunction:: medfix2
.. autofunction:: min_sampling
.. autofunction:: normalize_power
.. autofunction:: pad
.. autofunction:: pixelscale_nyquist
.. autofunction:: pv
.. autofunction:: rectangle
.. autofunction:: rms
.. autofunction:: shift
.. autofunction:: sin2
.. autofunction:: strehl
.. autofunction:: rebin
.. autofunction:: register
.. autofunction:: rescale
.. autofunction:: waffle2
.. autofunction:: zernike
.. autofunction:: zernike_basis
.. autofunction:: zernike_compose
.. autofunction:: zernike_fit
.. autofunction:: zernike_remove



