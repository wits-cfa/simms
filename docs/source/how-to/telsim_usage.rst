Telsim Usage
========================

Basic Simulation
------------------

To create a Measurement Set (MS), run::

    simms telsim --telescope meerkat --direction J2000,0.5deg,-31deg --start-time 2025-11-05T14:00:00 --dtime 2 --ntimes 100 --start-freq 1.0GHz --dfreq 1MHz --nchan 128 --correlation XX,XY,YX,YY example.ms

where:

- ``--telescope meerkat``: specifies the name of the array being used for the observation
- ``--direction J2000,0.5deg,-31deg``: specifies the pointing direction of the telescope in J200 reference frame.
- ``--starttime 2025-11-05T14:00:00``  : specifies the start time of the observation.
- ``--dtime 2``: specifies the time interval between consecutive integrations in seconds.
- ``--ntimes 100``: specifies the number of time integrations.
- ``--startfreq 1.0GHz``: specifies the starting frequency of the observation.
- ``--dfreq 1MHz``: specifies the frequency channel width.
- ``--nchan 128``: specifies the number of frequency channels.
- ``--correlations XX,YY,XY,YX``: specifies the polarization correlations to include in the MS. 
- ``example.ms``: the name of the output Measurement Set (MS) file. 


Customizing the array
----------------------

If you need to customize the array to use a different antenna layout, you can specify this using the sublist flag. For example::
    
   simms telsim --telescope skamid --direction J2000,0.5deg,-31deg --starttime 2025-11-05T14:00:00 --dtime 2 --ntimes 100 --startfreq 1.0GHz --dfreq 1MHz --nchan 128 --correlation XX,XY,YX,YY --subarray-list M001 M006 M009 M012 M034 custom_array.ms

where:
 - ``--sublist M001 M006 M009 M012 M034`` specifies the SKA Mid antennas to include in the subarray.

Another method to customize the array, though less intuitive, is to use the --subarray-range flag. This flag allows you to specify antennas by their indices in the array configuration. For example, to use antennas with indices 1, 3, 5, and 7, you would run::

    simms telsim --telescope skamid --direction J2000,0.5deg,-31deg --starttime 2025-11-05T14:00:00 --dtime 2 --ntimes 100 --startfreq 1.0GHz --dfreq 1MHz --nchan 128 --correlation XX,XY,YX,YY --subarray-range 1,7,2 custom_array.ms
 
where:
 - ``--subarray-range 1,7,2`` specifies the range of antennas to include in the subarray, starting from index 1 to index 7 with a step of 2. This would require you to know the indices of the antennas you wish to include. To know this, you can refer to the array configuration files located in the `simms/telescope/layouts` directory.

 A better approach to customizing the array is to give a yaml file with the list of antennas to use. For example, create a file called `my_array.yaml` with the following content::

    antnames:
      - M000
      - M001
      - M002
      - M003
      - M004

Then run the command::
      simms telsim --telescope skamid --direction J2000,0.5deg,-31deg --starttime 2025-11-05T14:00:00 --dtime 2 --ntimes 100 --startfreq 1.0GHz --dfreq 1MHz --nchan 128 --correlation XX,XY,YX,YY --subarray-file my_array.yaml custom_array.ms
   
where:
 - ``--subarray-file my_array.yaml`` specifies the yaml file containing the list of antennas to include in the subarray. The antennas specified must be part of the telescope given.

.. note::

   You can view the available telescope options by running::

      simms telsim --list
   

Adding noise
----------------

To add thermal noise to the MS, you can use the ``--sefd`` flag, which gives the System Equivalent Flux Density (SEFD) of the telescope. For example::

   simms telsim --telescope meerkat --direction J2000,0.5deg,-31deg --starttime 2025-11-05T14:00:00 --dtime 2 --ntimes 100 --startfreq 1.0GHz --dfreq 1MHz --nchan 128 --correlation XX,XY,YX,YY --sefd 420 example.ms

where:
- ``--sefd 420``: specifies the SEFD in Jy to use for adding thermal noise to the visibilities.

This will add the same thermal noise to all baselines and frequencies. If you want to specify different SEFDs for each antenna, you can provide a yaml file with the SEFD values ordered according to the antennas in the array configuration files. For example, create a file called `sefd_values.yaml` with the following content::

   sefd:
      -  400
      -  420
      -  450
      -  410
      -  430
      -  440
      -  415
Then run the command::

   simms telsim --telescope kat-7 --direction J2000,0.5deg,-31deg --starttime 2025-11-05T14:00:00 --dtime 2 --ntimes 100 --startfreq 1.0GHz --dfreq 1MHz --nchan 128  --sensitivity-file sefd_values.yaml example.ms

where:
- ``--sensitivity-file sefd_values.yaml``: specifies the yaml file containing the SEFD values for each antenna. In this case, we use the KAT-7 array, which has 7 antennas, hence 7 SEFD values are provided.

This will add thermal noise to the visibilities based on the SEFD values provided for each antenna. Each baseline will have noise calculated from the SEFDs of the two antennas forming that baseline.
.. note::

   The SEFD values provided must correspond to the antennas in the array configuration files. You can refer to the array configuration files located in the `simms/telescope/layouts` directory to know the order of antennas.

.. warning::

   When specifying the SEFD values in a yaml file, ensure that the number of SEFD values matches the number of antennas in the selected telescope array. Mismatched numbers will lead to errors during simulation.

It is possible to specify frequency-dependent SEFD values by providing a list of SEFDs with the corresponding frequencies in the yaml file. For example, create a file called `freq_sefd_values.yaml` with the following content::

   freq: [1.0GHz, 1.2GHz, 1.4GHz, 1.6GHz]
   sefd: [400, 350, 300, 450]
Then run the command::

   simms telsim --telescope meerkat --direction J2000,0.5deg,-31deg --starttime 2025-11-05T14:00:00 --dtime 2 --ntimes 100 --startfreq 1.0GHz --dfreq 1MHz --nchan 128  --sensitivity-file freq_sefd_values.yaml --smooth spline --fit-order 6 example.ms

where:
- ``--sensitivity-file freq_sefd_values.yaml``: specifies the yaml file containing the frequency-dependent SEFD values.
- ``--smooth spline``: specifies the smoothing method to use when interpolating SEFD values across frequencies. Options include 'spline' and 'polyn'.
- ``--fit-order 6``: specifies the order of the fit.

This will add thermal noise to the visibilities based on the frequency-dependent SEFD values provided for each antenna. The SEFD values will be interpolated to match the frequency channels of the observation.
.. note::

   When specifying frequency-dependent SEFD values, ensure that the frequencies provided cover the range of observation frequencies for a good fit.


Flagging data based on source elevation
------------------------------------------
You can flag data based on the elevation of the source using the ``--low-elevation-limit`` and ``--high-elevation-limit`` flags. For example::

   simms telsim --telescope meerkat --direction J2000,0.5deg,-31deg --starttime 2025-11-05T14:00:00 --dtime 2 --ntimes 100 --startfreq 1.0GHz --dfreq 1MHz --nchan 128 --correlation XX,XY,YX,YY --low-elevation-limit 15 --high-elevation-limit 80 example.ms

where:
- ``--low-elevation-limit 15``: specifies the minimum elevation in degrees below which data will be flagged.
- ``--high-elevation-limit 80``: specifies the maximum elevation in degrees above which data will be flagged.
