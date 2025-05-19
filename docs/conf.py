# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#path-setup

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "soogo (Surrogate-based 0-th Order Global Optimization)"
copyright = "2025, Alliance for Sustainable Energy, LLC"
author = "Weslley S. Pereira"
release = "1.2.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # This is for automatic gen of rst and other things
    "sphinx_autodoc_typehints",  # Including typehints automatically in the docs
    "sphinx.ext.mathjax",  # This is for LaTeX
]

# General config
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# sphinx.ext.autodoc
autodoc_default_options = {
    "special-members": "__call__",
    "exclude-members": "set_predict_request, set_score_request",
}

# sphinx_autodoc_typehints
typehints_use_signature = True
typehints_use_signature_return = True
typehints_defaults = "braces-after"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
