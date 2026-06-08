"""Sphinx configuration for TimeFeatures."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

project = "TimeFeatures"
author = "Alejandro Rivas Garcia"
copyright = "2026, Alejandro Rivas Garcia"
release = "1.0.18"

extensions = []
templates_path = ["_templates"]
exclude_patterns = ["build", "Thumbs.db", ".DS_Store"]

language = "en"
html_theme = "alabaster"
html_static_path = []
html_title = "TimeFeatures documentation"
html_short_title = "TimeFeatures"

rst_prolog = """
.. |addon| replace:: TimeFeatures
"""
