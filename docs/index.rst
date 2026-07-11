simms
=====

**simms** simulates radio-interferometer observations end to end: ``telsim``
builds a `Measurement Set (MS) <https://casa.nrao.edu/Memos/229.html>`_ from
a telescope layout, and ``skysim`` predicts model visibilities from a sky
model (ASCII catalogue, FITS image, or WSClean component list) into it. A
standalone ``primary-beam`` tool exposes the beam operations on their own.

.. code-block:: console

    $ simms telsim --telescope kat-7 --startfreq 900MHz --dfreq 1MHz --nchan 64 obs.ms
    $ simms skysim --ascii-sky skymodel.txt --column DATA obs.ms

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: Concepts

   concepts/telsim
   concepts/skysim
   concepts/schemas
   concepts/ms-conventions

.. toctree::
   :maxdepth: 2
   :caption: Using simms

   cli

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Project

   contributing


Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
