"""Sphinx configuration for TimeFeatures."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# -- Project information --------------------------------------------------
project = "TimeFeatures"
author = "Alejandro Rivas Garcia"
copyright = "2026, Alejandro Rivas Garcia"

# Read version from the single source of truth.
_version_globals = {}
exec((ROOT / "timefeatures" / "__version__.py").read_text(encoding="utf-8"), _version_globals)
release = _version_globals["__version__"]
version = ".".join(release.split(".")[:2])

# -- General configuration ------------------------------------------------
extensions = [
    # Auto-generate :ref: labels from section titles so we can write
    # `:ref:`Edge Weights`` instead of declaring labels by hand.
    "sphinx.ext.autosectionlabel",
    # Link to Orange / NumPy / SciPy reference docs.
    "sphinx.ext.intersphinx",
]

# Prefix labels with the document name so two pages can share the same
# section title (e.g. "Usage Example") without clashing.
autosectionlabel_prefix_document = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "orange": ("https://orange3.readthedocs.io/projects/orange-data-mining-library/en/latest/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

language = "en"

# -- HTML output ----------------------------------------------------------
html_theme = "alabaster"
html_static_path = []
html_title = "TimeFeatures documentation"
html_short_title = "TimeFeatures"
html_theme_options = {
    "description": (
        "Time-series feature engineering for Orange3: build derived "
        "variables, visualise their dependencies, and persist data to "
        "PostgreSQL or MySQL."
    ),
    "github_user": "alervgr",
    "github_repo": "Orange-TimeFeatures",
    "github_banner": True,
    "show_powered_by": False,
    "fixed_sidebar": True,
}

# Default replacements for any RST file (no need to redeclare per page).
rst_prolog = """
.. |addon| replace:: TimeFeatures
"""
