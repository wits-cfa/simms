.. _telsim:

telsim -- building a Measurement Set
=====================================

``telsim`` builds a simulated `Measurement Set (MS)
<https://casa.nrao.edu/Memos/229.html>`_ from a telescope layout: it lays out
antennas, computes ``uvw`` for the requested time/frequency grid, and writes
the standard MS subtables (``ANTENNA``, ``FIELD``, ``SPECTRAL_WINDOW``,
``POINTING``, ...) with no visibility data yet -- that's what :doc:`skysim`
fills in afterwards.

.. contents::
   :local:
   :depth: 2

Required inputs
----------------

- **ms**: the name of the MS to create.
- **telescope**: the telescope array layout (see ``simms telsim --list`` for
  the bundled layouts under ``src/simms/telescope/layouts/``).
- **direction**: the pointing direction, written to ``FIELD.PHASE_DIR`` /
  ``POINTING.DIRECTION`` (see :doc:`ms-conventions` for why these differ).
- **starttime**, **dtime**, **ntimes**: the observation's time grid.
- **startfreq**, **dfreq**, **nchan**: the observation's frequency grid.

Usage
-----

.. code-block:: console

    $ simms telsim --telescope kat-7 --direction "J2000,0h24m20s,-30d12m33s" \
        --starttime 2024-03-14T06:15:10 --dtime 8 --ntime 100 \
        --startfreq 900MHz --dfreq 1MHz --nchan 64 obs.ms

List the available telescope layouts:

.. code-block:: console

    $ simms telsim --list

See :doc:`../cli` for the full option reference.

Where to next
-------------

- :doc:`skysim` -- simulate visibilities into the MS ``telsim`` just created.
- :doc:`ms-conventions` -- how simms reads/writes MS metadata, and the
  distinction between phase centre and pointing centre.
