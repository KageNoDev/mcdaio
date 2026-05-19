"""Sphinx configuration for the mcdaio docs."""

from __future__ import annotations

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath("../.."))

project = "mcdaio"
author = "Laura Białobrzewska"
copyright = f"{date.today().year}, {author}"
release = "0.2.1"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

autodoc_mock_imports = ["streamlit", "plotly", "pandas"]
