Command-line interface
========================

The ``simms`` command is a click group chaining three subcommands:
``telsim``, ``skysim``, and ``primary-beam``. Global options (``--ms``,
``--log-level``, ``--chain``) precede the subcommand:

.. code-block:: console

    $ simms [--log-level LEVEL] [--ms FILE --chain] COMMAND ...

Pass ``--chain`` on the group to run ``telsim`` and ``skysim`` back to back
against one MS given once at the top level (``--ms`` is then dropped from the
subcommands):

.. code-block:: console

    $ simms --ms obs.ms --chain telsim --telescope kat-7 skysim --ascii-sky sky.txt

.. click:: simms.apps.main:cli
   :prog: simms
   :nested: full
