# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#path-setup

import os
import sys
import re

sys.path.insert(0, os.path.abspath(".."))


def setup(app):
    app.connect("html-page-context", add_version_context)


def add_version_context(app, pagename, templatename, context, doctree):
    if "versions" not in context:
        return

    versions = context["versions"]
    tags = []
    branches = []

    for version in versions:
        item = {
            "name": version.name,
            "url": version.url,
        }

        if re.match(smv_tag_whitelist, version.name):
            tags.append(item)
        else:
            branches.append(item)

    context["versions"] = {
        "tags": sorted(tags, key=lambda x: x["name"], reverse=True),
        "branches": branches,
    }


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "soogo (Surrogate-based 0-th Order Global Optimization)"
copyright = "2025, Alliance for Sustainable Energy, LLC"
author = "Weslley S. Pereira"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",  # This is for automatic gen of rst and other things
    "sphinx.ext.autosummary",  # This is for automatic summary tables
    "sphinx_autodoc_typehints",  # Including typehints automatically in the docs
    "sphinx.ext.mathjax",  # This is for LaTeX
    "myst_parser",  # This is for markdown support
    "sphinx_multiversion",  # This is for multiple version support
]

# General config
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# sphinx.ext.autodoc
autodoc_default_options = {
    "special-members": "__call__",
    "exclude-members": "set_predict_request, set_score_request",
}

# sphinx.ext.autosummary
autosummary_generate = True

# sphinx_autodoc_typehints
typehints_use_signature = True
typehints_use_signature_return = True
typehints_defaults = "braces-after"

# myst_parser
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# sphinx_multiversion
smv_branch_whitelist = r"^main$"
smv_latest_version = "main"
smv_tag_whitelist = r"^v\d+\.\d+\.\d+$"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_context = {
    "display_github": True,
    "github_user": "NREL",
    "github_repo": "soogo",
    "github_version": "main",
    "conf_py_path": "/docs/",
}


# Copy markdown structured README from root to docs
markdown_files = ["README.md", "LICENSE", "CONTRIBUTING.md"]
for md_file in markdown_files:
    src = os.path.abspath(os.path.join("..", md_file))
    dst = os.path.abspath(os.path.join(".", md_file))
    if os.path.exists(src):
        import shutil
        from pathlib import Path

        shutil.copyfile(src, dst)

        text = Path(src).read_text()
        for old, new in {
            "](soogo/": "](../soogo/",
            "](examples/": "](../examples/",
            "](tests/": "](../tests/",
            "](pyproject.toml": "](../pyproject.toml",
        }.items():
            text = text.replace(old, new)

        Path(dst).write_text(text)
