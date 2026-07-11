.. _skysim:

skysim -- predicting visibilities
===================================

``skysim`` predicts model visibilities from a sky model (ASCII catalogue,
FITS image(s), or a WSClean component list) into an existing MS, writing them
to a chosen data column. It can also add or subtract an existing column
in-place instead of simulating from a sky model.

Basic usage
-----------

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column DATA visdata.ms

- ``--ascii-sky skymodel.txt`` -- a catalogue of parametrised sources.
- ``--column DATA`` -- the column where simulated visibilities are written.
- ``visdata.ms`` -- the target MS, which must already exist (from an
  observation, or created with :doc:`telsim`).

ASCII sky model schema
-----------------------

- **Point sources** need only RA, Dec, and intensity (``stokes_i``).
- **Extended sources** are 2D Gaussians, parametrised by FWHM major/minor axes
  (``emaj``/``emin``) and position angle (``pa``); double-horn profiles are
  not supported.
- **Spectral line sources** need the peak frequency (``line_peak``) and width
  (``line_width``).
- **Continuum sources** need a reference frequency (``cont_reffreq``) and at
  least one power-law coefficient (``cont_coeff_1`` = spectral index,
  ``cont_coeff_2`` = curvature, ...).

See :doc:`schemas` for the full schema, and use ``--ascii-species`` to select
a non-default catalogue mapping (e.g. ``bdsf_gaul`` for a PyBDSF catalogue).

FITS sky models
----------------

.. code-block:: console

    $ simms skysim --fits-sky skymodel.fits --column DATA visdata.ms

Provide separate FITS files per Stokes when simulating polarised sources.
Tune the prediction with ``--pixel-tol`` (minimum pixel brightness considered,
default ``1e-7``), ``--fft-precision`` (``single``/``double``), and
``--no-do-wstacking`` to disable w-stacking.

Adding or subtracting an existing column
------------------------------------------

Once visibilities are simulated into one column, add or subtract them
against another:

.. code-block:: console

    $ simms skysim --ic DATA --column MODEL_DATA --mode add visdata.ms
    $ simms skysim --ic DATA --column MODEL_DATA --mode subtract visdata.ms

``--ic``/``--input-column`` is the source column; ``--column`` is where the
result is written; ``--mode`` defaults to ``simulate``.

Thermal noise
-------------

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column SIMULATED_DATA --sefd 421 visdata.ms

Provide either ``--sefd`` (System Equivalent Flux Density, in Jy) or
``--tsys-over-eta`` (:math:`T_\mathrm{sys}/\eta`).

Chunking large MSs
-------------------

.. code-block:: console

    $ simms skysim --ascii-sky skymodel.txt --column SIMULATED_DATA --row-chunks 5000 largevis.ms

``--row-chunks`` controls the row-wise task/memory granularity (default
``10000``).

Where to next
-------------

- :doc:`telsim` -- create the target MS.
- :doc:`ms-conventions` -- how the primary beam centre is read from
  ``POINTING.DIRECTION``.
- :doc:`schemas` -- full sky model and catalogue-mapper schemas.
- :doc:`../cli` -- full option reference, including all abbreviations.
