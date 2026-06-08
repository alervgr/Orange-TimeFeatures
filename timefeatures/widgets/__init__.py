DESCRIPTION = "Time features tools for Orange."

BACKGROUND = "#5bcebf"

ICON = "icons/timefeature-xs.svg"


def _prefer_orange_help_window():
    """Open add-on help in Orange's Help dock instead of a web browser."""
    try:
        from AnyQt.QtCore import QSettings
    except Exception:
        return
    QSettings().setValue("help/open-in-external-browser", False)


_prefer_orange_help_window()
