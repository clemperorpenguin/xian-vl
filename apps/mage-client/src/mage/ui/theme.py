# MAGE — Gaming HUD for real-time screen translation.
# Copyright (C) 2026  Clementine Pendragon <clem@pendragon.systems>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# Contact: clem@pendragon.systems (Clementine Pendragon, c/o Xian Project Development)

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
