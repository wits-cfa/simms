.. _telsim_tutorial:

Array Telescope Simulations (``telsim``)
========================================

.. contents::
   :local:
   :depth: 2


Inputs Required
---------------

- **ms**: The name of the MS to create.
- **telescope**: The telescope configuration.
- **direction**: The pointing direction.
- **starttime**: The start time of the observation.
- **dtime**: The time interval between observations.
- **ntimes**: The number of time steps.
- **startfreq**: The starting frequency.
- **dfreq**: The frequency interval.
- **nchan**: The number of frequency channels.

Steps to Use ``telsim``
-----------------------

1. Provide the required inputs (e.g., MS name, telescope configuration, etc.).
2. Run ``telsim`` with the desired parameters.
3. Verify that the MS has been created successfully.

Best Practices
--------------

- Choose a telescope configuration that matches your simulation goals.
- Ensure that the time and frequency ranges are realistic for your observation.

``telsim`` is often used in combination with ``skysim`` to simulate visibilities based on a sky model.
