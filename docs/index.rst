**prtools** --- Utility functions for image-based phase retrieval
=================================================================

**Version**: |version|

**Useful links**:
`Source Repository <https://github.com/andykee/prtools>`_ |
`Releases <https://pypi.org/project/prtools/>`_ |
`Issue Tracker <https://github.com/andykee/prtools/issues>`_ 

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
   
   rms
   pv
   radial_avg
   ee

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
