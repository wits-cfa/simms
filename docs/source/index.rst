.. simms documentation master file, created by
   sphinx-quickstart on Mon Oct 24 14:11:23 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Simms 3.0
===========

Introduction
------------

**simms** is a simulation framework built around two key tools: ``telsim`` and ``skysim``:

- ``telsim``: Creates simulated Radio Interferometry Array data (visibilities) into `MS files <https://casa.nrao.edu/Memos/229.html>`_
- ``skysim``: Populates this MS by simulating visibilities, given a sky model. ``skysim`` also has an additional feature that lets you add or subtract from an existing data column.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   how-to/installation
   how-to/usage
   tutorials/tutorials
   reference/reference

.. Indices and tables
.. ==================

.. * :ref:`genindex`
.. * :ref:`modindex`
.. * :ref:`search`