.. _installation:

Installation
============

Requirements
------------

* Python 3.11 -- 3.13.

From PyPI
---------

The stable release is available on `PyPI <https://pypi.org/project/simms>`_:

.. code-block:: console

    $ pip install "simms>=3.0"

From GitHub
-----------

To install the latest development version:

.. code-block:: console

    $ pip install git+https://github.com/wits-cfa/simms.git

Either way, this installs the ``simms`` command-line tool and the importable
``simms`` package.

For development
----------------

The project uses `uv <https://docs.astral.sh/uv/>`_ -- see it for
everything, rather than calling ``python``/``pytest``/``ruff`` directly:

.. code-block:: console

    $ git clone https://github.com/wits-cfa/simms.git
    $ cd simms
    $ uv run --group tests python -m pytest
    $ uv run --group ruff ruff check src tests

To build the documentation locally:

.. code-block:: console

    $ uv sync --group docs
    $ uv run sphinx-build -b html docs docs/_build/html
    $ open docs/_build/html/index.html
