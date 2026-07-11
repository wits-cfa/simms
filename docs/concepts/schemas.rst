.. _schemas:

Schemas
=======

Sky model and cab schemas are declared as YAML files under
``src/simms/schemas/`` and used to validate/clickify CLI parameters (see
:doc:`../cli`).

simms sky model schema
-----------------------

.. literalinclude:: ../../src/simms/schemas/source_schema.yaml
    :language: yaml


PyBDSF sky model schema mapper (Gaussian source list; Gaul)
-------------------------------------------------------------

.. literalinclude:: ../../src/simms/schemas/bdsf_gaul_source_mapper.yaml
    :language: yaml
