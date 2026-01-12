# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
from datetime import date

from simms import PCKGDIR, __version__

author = "Sphesihle Makhathini, Mika Naidoo, Mukundi Ramanyimi,"
" Shibre Semane, Galefang Mapunda, Senkhosi Simelane"
project = "simms"
copyright = f"{date.today().year}, {author}"

release = version = __version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinxcontrib.autoyaml",
    #    'sphinx_rtd_theme',
]

sys.path.insert(0, PCKGDIR)


templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "classic"
