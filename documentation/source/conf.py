import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join('..', '..')))
# -- Project information -----------------------------------------------------

project = 'tspredictor'
copyright = '2021, rsc18 and ci20l'
author = 'rsc18 and ci20l'

# The full version, including alpha/beta/rc tags
release = '0.0.1'

# -- Extensions --------------------------------------------------------------
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# -- Options for HTML output -------------------------------------------------

html_title = project

# NOTE: All the lines are after this are the theme-specific ones. These are
#       written as part of the site generation pipeline for this project.
# !! MARKER !!
html_theme = "sphinx_rtd_theme"
