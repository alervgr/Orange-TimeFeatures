"""Check that an installed TimeFeatures (not the source checkout) ships
everything Orange needs: the widget and help entry points, importable
widget modules, their icons and the bundled in-app help."""
import importlib
import sys
from importlib.metadata import entry_points
from pathlib import Path

import timefeatures
from timefeatures.help import WIDGET_HELP_PATH

WIDGET_MODULES = (
    "owtimefeaturesconstructor",
    "owvardependencygraph",
    "owsavetodb",
    "owloadfromdb",
)

errors = []
package_dir = Path(timefeatures.__file__).resolve().parent
if "site-packages" not in package_dir.parts:
    errors.append(f"imported from {package_dir}, not from site-packages")

if not any(ep.value == "timefeatures.widgets"
           for ep in entry_points(group="orange.widgets")):
    errors.append("missing 'orange.widgets' entry point")
if not any(ep.value.startswith("timefeatures.help:")
           for ep in entry_points(group="orange.canvas.help")):
    errors.append("missing 'orange.canvas.help' entry point")

for name in WIDGET_MODULES:
    module = importlib.import_module(f"timefeatures.widgets.{name}")
    widget = next(
        (obj for obj in vars(module).values()
         if isinstance(obj, type) and obj.__module__ == module.__name__
         and getattr(obj, "icon", None)),
        None,
    )
    if widget is None:
        errors.append(f"{name}: no widget class found")
    elif not (package_dir / "widgets" / widget.icon).is_file():
        errors.append(f"{name}: icon {widget.icon} not installed")

help_index = Path(WIDGET_HELP_PATH[0][0])
if not help_index.is_file():
    errors.append(f"bundled help missing: {help_index}")

if errors:
    sys.exit("Installed package is incomplete:\n  " + "\n  ".join(errors))
print(f"OK: TimeFeatures installed at {package_dir}")
