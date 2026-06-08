"""Orange Canvas help integration for TimeFeatures.

Orange resolves widget help through the ``orange.canvas.help`` entry point.
The HTML index provider reads the Sphinx-generated index and maps widget
names from the ``widgets`` section to their documentation pages.
"""

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
HELP_XPATH = ".//*[@id='widgets']//li/a"

WIDGET_HELP_PATH = (
    (
        str(PACKAGE_ROOT / "help_html" / "index.html"),
        HELP_XPATH,
    ),
    (
        str(PROJECT_ROOT / "docs" / "build" / "html" / "index.html"),
        HELP_XPATH,
    ),
    (
        "{DATA_DIR}/share/help/en/timefeatures/index.html",
        HELP_XPATH,
    ),
)
