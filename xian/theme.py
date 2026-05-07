"""System accent color utilities.

Reads the user's system accent color from the Qt palette and provides it
as hex strings for use in stylesheets and QPen/QColor constructors.
"""

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor


def _system_accent() -> QColor:
    """Return the system accent QColor, falling back to a teal accent."""
    app = QApplication.instance()
    if app is not None:
        color = app.palette().color(QPalette.ColorRole.Accent)
        # Qt returns a valid color even if the platform has no accent;
        # on some desktops it falls back to QPalette::Highlight instead.
        if color.isValid() and color != QColor(0, 0, 0):
            return color
        # Fallback: try Highlight role (older Qt / GTK desktops)
        color = app.palette().color(QPalette.ColorRole.Highlight)
        if color.isValid() and color != QColor(0, 0, 0):
            return color
    # Ultimate fallback — a pleasant teal that looks good on dark UIs
    return QColor("#26A69A")


def accent_hex() -> str:
    """Return the accent colour as a ``#RRGGBB`` hex string."""
    return _system_accent().name()


def accent_hover_hex() -> str:
    """Return a slightly darker shade for hover states."""
    c = _system_accent().darker(115)
    return c.name()


def accent_qcolor(alpha: int = 255) -> QColor:
    """Return the accent as a QColor with optional alpha."""
    c = _system_accent()
    c.setAlpha(alpha)
    return c
