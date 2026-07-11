"""Sphinx configuration for the simms documentation.

Autodoc imports the ``simms`` package, so the build environment must have it
installed (``uv sync --group docs`` locally; Read the Docs installs it via
``.readthedocs.yaml``). The package lives under ``src/``, added to sys.path
below so an editable/uninstalled checkout also builds.
"""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../src"))

from simms import __version__  # noqa: E402

# -- Project information -----------------------------------------------------

project = "simms"
author = "Sphesihle Makhathini, Mika Naidoo, Mukundi Ramanyimi, Shibre Semane, Galefang Mapunda, Senkhosi Simelane"
copyright = f"{date.today().year}, {author}"

version = __version__
release = __version__

# -- General configuration ---------------------------------------------------

language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "sphinx_click",
    "sphinxcontrib.autoyaml",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

nitpicky = False

# -- Autodoc -------------------------------------------------------------

autodoc_member_order = "bysource"
autodoc_typehints = "description"

napoleon_google_docstring = True
napoleon_numpy_docstring = True

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

# -- HTML output -------------------------------------------------------------

html_theme = "furo"
html_title = f"simms {release}"
html_static_path = ["_static"]

html_theme_options = {
    "source_repository": "https://github.com/wits-cfa/simms/",
    "source_branch": "main",
    "source_directory": "docs/",
}

# -- MyST (markdown) ---------------------------------------------------------

myst_enable_extensions = ["colon_fence", "deflist"]
