.. _skysim_tutorial:

Sky Simulations (``skysim``)
========================================

.. contents::
   :local:
   :depth: 2

Steps to Use ``skysim``:
--------------------------

1. Provide an MS 
   - [either existing or created with ``telsim``]
2. Provide a sky model input 
   - Either an ascii file from a catalogue or 
   - A fits image
3. Run ``skysim`` with your desired parameters
   - ``skysim`` will compute your visibilities and write them to the specified column in the MS
4. Optionally:
    - add/subtract data 
    - add noise 


Best Practices
-----------------

- Provide separate fits files for each Stokes when simulating polarized sources
- Use custom mapping files for non-standard catalogue formats
- Ensure frequency ranges match between your fits images and MS when using fits as model input


Command-Line Abbreviations
--------------------------

Refer to ``simms-cab.yaml`` for a full list of abbreviations.

+--------+-----------------+
| Abbrev | Full Option     |
+========+=================+
|| -as   || --ascii-sky    |
|| -fs   || --fits-sky     |
|| -col  || --column       |
|| -ic   || --input-column |
+--------+-----------------+
