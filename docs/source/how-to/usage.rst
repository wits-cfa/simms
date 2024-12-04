.. _usage:
.. role:: raw-math(raw)
    :format: latex html

Running Simms 3.0
==================
Simms 3.0 has to applications (or scripts). The first ``telsim`` is for generating simulated radio interferometry data sets (or visibilities). This application produces an empty `Measurement Set (MS) <https://casa.nrao.edu/Memos/229.html>`_ given an array telescope's antenna layout and observation parameters. Use ``telsim --help`` for a general overview of the application and defintion of parameters. The package includes these known telescope antenna layouts:

.. list-table:: Array telescope layouts available in Simms
   :widths: 40 50
   :header-rows: 1

   * - Telescope
     - Simms 3.0 label
   * - MeerKAT
     - meerkat
   * - KAT 7
     - kat-7
   *
     - WSRT 
     - wsrt
   * - VLA
     - vla-a, vla-b, vla-c, vla-d

Say, we wanted to simulate a 30 minute MeerKAT observation at 1.4 GHz. Let's also set the exposure time to 5 seconds, the channel width to 1~MHz, and set observing direction to RA=23h59m0s and Dec=-30d0m0s. The command to run in this case is:

.. code-block:: bash

    telsim --telescope meerkat --start-freq 1.4GHz --dfreq 1MHz --direction J2000,23h59m0s,-30d0m0s --dtime 5 --ntime  360 my-meerkat-obs.ms

Note that ``--ntimes`` is the number of exposures that make up the observation time period i.e, :raw-math:`$$\frac{30 \times 60}{5} = 360$$` exposures. The output MS is ``my-meerkat-obs.ms``

The other application is ``skysim``. This simulates a catalogue of sources (or sky model) into an MS. For example, I can simulate a sky model defined through a file ``my-skymodel.txt`` into the ``my-meerkat-obs.ms`` MS from above by running:
   
.. code-block:: bash

    skysim --catalogue my-skymodel.txt my-meerkat-obs.ms
