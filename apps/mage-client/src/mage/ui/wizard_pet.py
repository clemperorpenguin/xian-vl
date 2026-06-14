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

"""Desktop Wizard — a floating pixel companion that walks the bottom of the screen.

The wizard is the "chat mode equivalent" surfaced as a living desktop pet: it
wanders along the screen floor, dodges the cursor, reacts when a translation is
running, and opens the Chat sidebar when clicked. It does *no* inference itself —
every state change is driven by signals the app already emits, and all animation
runs off plain ``QTimer`` ticks on the GUI thread, so it never touches the
inference hot path.

Art is fully optional. Drop sprite frames into ``wizard/`` at the workspace root
named ``<state>_<n>.png`` (e.g. ``idle_0.png``, ``walk_0.png``, ``cast_0.png``,
``react_0.png``, ``sad_0.png``) and they are picked up automatically. Until then,
a vector placeholder wizard is drawn so the whole system is testable today.
"""

import logging
import os
import random
import time
from enum import Enum, auto

from PyQt6.QtWidgets import QWidget, QLabel, QMenu, QVBoxLayout, QApplication
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPixmap, QPolygon, QCursor, QGuiApplication,
    QFont, QPainterPath,
)

from mage.ui.overlay_base import MageOverlayWindow
from mage.ui.theme import accent_hex
from mage.resources import get_resource_path
from shared_types.state import t

logger = logging.getLogger(__name__)

# --- tunables ---------------------------------------------------------------
SPRITE_W = 96
SPRITE_H = 96
FLOOR_MARGIN = 6          # px above the screen's bottom edge
WALK_STEP = 4             # px advanced per behaviour tick while walking
ANIM_INTERVAL_MS = 140    # frame-swap cadence
TICK_INTERVAL_MS = 33     # behaviour / movement cadence (~30 fps)
AVOID_RADIUS = 150        # cursor proximity that triggers a teleport-flee
DRAG_THRESHOLD = 6        # px of movement before a press becomes a drag
CAST_HOLD_S = 2.0         # how long a "thinking" pose lingers between updates
REACT_HOLD_S = 3.0        # how long the wizard celebrates a finished result
SAD_HOLD_S = 2.5
DWELL_MIN_S = 1.5         # idle pause range between walks
DWELL_MAX_S = 4.5
BUBBLE_MS = 4500          # speech-bubble lifetime


class WizardState(Enum):
    IDLE = auto()
    WALK = auto()
    CAST = auto()    # a translation is running
    REACT = auto()   # a translation just finished
    SAD = auto()     # a translation errored


# Placeholder palette per state (used only until real art is dropped in).
_STATE_TINT = {
    WizardState.IDLE: QColor(80, 130, 220),
    WizardState.WALK: QColor(90, 160, 235),
    WizardState.CAST: QColor(150, 90, 230),
    WizardState.REACT: QColor(80, 200, 130),
    WizardState.SAD: QColor(130, 130, 140),
}


class WizardSpeechBubble(MageOverlayWindow):
    """Tiny auto-closing speech bubble shown above the wizard."""

    def __init__(self, app=None, parent=None):
        super().__init__(window_id="wizard_bubble", app=app, parent=parent)
        # A speech bubble should never be draggable or steal interaction.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self._label = QLabel(self)
        self._label.setWordWrap(True)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._label.setStyleSheet(
            f"color: #f4f4f4; background: rgba(20,20,24,235);"
            f"border: 1px solid {accent_hex()}; border-radius: 10px;"
            f"padding: 8px 10px; font-size: 13px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)
        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.hide)

    def say(self, text: str, anchor: QRect, ms: int = BUBBLE_MS):
        text = (text or "").strip()
        if not text:
            return
        if len(text) > 140:
            text = text[:139].rstrip() + "…"
        self._label.setText(text)
        self.setFixedWidth(260)
        self._label.adjustSize()
        self.adjustSize()
        # Anchor centred above the wizard, clamped onto its screen.
        x = anchor.center().x() - self.width() // 2
        y = anchor.top() - self.height() - 6
        screen = (QGuiApplication.screenAt(anchor.center())
                  or QGuiApplication.primaryScreen())
        geo = screen.availableGeometry()
        x = max(geo.left() + 4, min(x, geo.right() - self.width() - 4))
        if y < geo.top() + 4:
            y = anchor.bottom() + 6
        self.move(x, y)
        self.show()
        self.raise_()
        self._close_timer.start(ms)


class WizardPet(MageOverlayWindow):
    """A frameless, always-on-top sprite that lives on the desktop floor."""

    def __init__(self, app=None, parent=None):
        super().__init__(window_id="wizard_pet", app=app, parent=parent)
        self.setFixedSize(SPRITE_W, SPRITE_H)

        # --- animation / behaviour state ---
        self._state = WizardState.IDLE
        self._behaviour = WizardState.IDLE      # autonomous mode: IDLE or WALK
        self._facing = 1                        # +1 faces right, -1 faces left
        self._frame = 0
        self._target_x = self.x()
        now = time.monotonic()
        self._dwell_until = now + random.uniform(DWELL_MIN_S, DWELL_MAX_S)
        self._emote_until = 0.0                 # while > now, an emote overrides autonomy

        # --- drag vs click discrimination ---
        self._press_global = QPoint()
        self._press_origin = QPoint()
        self._dragging_user = False

        # --- frames: real art if present, else vector placeholder ---
        self._frames: dict[WizardState, list[QPixmap]] = {}
        self._load_all_frames()

        self._bubble = WizardSpeechBubble(app=self.app, parent=self)

        self._snap_to_floor()

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._on_behaviour_tick)

    # ----- asset loading ----------------------------------------------------
    def _load_all_frames(self):
        for st in WizardState:
            self._frames[st] = self._load_frames(st.name.lower())

    def _load_frames(self, state_name: str) -> list[QPixmap]:
        frames: list[QPixmap] = []
        i = 0
        while True:
            rel = f"wizard/{state_name}_{i}.png"
            path = get_resource_path(rel)
            if path == rel or not os.path.exists(path):
                break
            pm = QPixmap(path)
            if pm.isNull():
                break
            frames.append(pm.scaled(
                SPRITE_W, SPRITE_H,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            ))
            i += 1
        return frames

    # ----- public API (driven by the app's inference signals) ---------------
    def on_thinking(self):
        """A translation has started; hold a casting pose."""
        self._emote_until = time.monotonic() + CAST_HOLD_S
        self._set_state(WizardState.CAST)

    def on_result(self, text: str):
        """A translation finished; celebrate and pop a speech bubble."""
        self._emote_until = time.monotonic() + REACT_HOLD_S
        self._set_state(WizardState.REACT)
        if text and text.strip():
            self._bubble.say(text, self.geometry())

    def on_error(self):
        self._emote_until = time.monotonic() + SAD_HOLD_S
        self._set_state(WizardState.SAD)
        self._bubble.say("?!", self.geometry(), ms=2000)

    def shutdown(self):
        """Stop all timers and dismiss the bubble (called when disabling)."""
        self._anim_timer.stop()
        self._tick_timer.stop()
        if self._bubble:
            self._bubble.close()

    # The wizard must always be clickable, so block the click-through plumbing
    # the binder may apply to ordinary overlays.
    def set_click_through(self, click_through: bool):
        self._click_through = False
        if not self._edit_mode_active:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    # ----- lifecycle --------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        self._snap_to_floor()
        self._anim_timer.start(ANIM_INTERVAL_MS)
        self._tick_timer.start(TICK_INTERVAL_MS)

    def hideEvent(self, event):
        self._anim_timer.stop()
        self._tick_timer.stop()
        if self._bubble:
            self._bubble.hide()
        super().hideEvent(event)

    # ----- animation --------------------------------------------------------
    def _set_state(self, state: WizardState):
        if state != self._state:
            self._state = state
            self._frame = 0
            self.update()

    def _on_anim_tick(self):
        frames = self._frames.get(self._state) or []
        n = len(frames) if frames else 2  # placeholder bobs over 2 phases
        self._frame = (self._frame + 1) % max(1, n)
        self.update()

    # ----- behaviour / movement --------------------------------------------
    def _home_geometry(self) -> QRect:
        center = self.geometry().center()
        screen = QGuiApplication.screenAt(center) or QGuiApplication.primaryScreen()
        return screen.availableGeometry()

    def _floor_y(self, home: QRect) -> int:
        return home.bottom() - self.height() - FLOOR_MARGIN

    def _snap_to_floor(self):
        home = self._home_geometry()
        y = self._floor_y(home)
        x = max(home.left(), min(self.x(), home.right() - self.width()))
        self.move(x, y)

    def _on_behaviour_tick(self):
        if self._dragging_user:
            return
        now = time.monotonic()

        # An active emote (cast/react/sad) freezes wandering until it expires.
        if now < self._emote_until:
            return
        if self._state in (WizardState.CAST, WizardState.REACT, WizardState.SAD):
            self._behaviour = WizardState.IDLE
            self._dwell_until = now + random.uniform(DWELL_MIN_S, DWELL_MAX_S)
            self._set_state(WizardState.IDLE)

        home = self._home_geometry()
        y = self._floor_y(home)

        # Dodge the cursor: if it gets too close, teleport to the far side.
        cursor = QCursor.pos()
        center = QPoint(self.x() + self.width() // 2, self.y() + self.height() // 2)
        if (abs(cursor.x() - center.x()) < AVOID_RADIUS
                and abs(cursor.y() - center.y()) < AVOID_RADIUS):
            self._flee(cursor, home, y)
            return

        if self._behaviour == WizardState.WALK:
            cur_x = self.x()
            dx = self._target_x - cur_x
            if abs(dx) <= WALK_STEP:
                self.move(self._target_x, y)
                self._behaviour = WizardState.IDLE
                self._dwell_until = now + random.uniform(DWELL_MIN_S, DWELL_MAX_S)
                self._set_state(WizardState.IDLE)
            else:
                self._facing = 1 if dx > 0 else -1
                self.move(cur_x + (WALK_STEP if dx > 0 else -WALK_STEP), y)
                self._set_state(WizardState.WALK)
        else:  # IDLE
            self.move(self.x(), y)  # keep glued to the floor if the screen changed
            self._set_state(WizardState.IDLE)
            if now >= self._dwell_until:
                lo = home.left()
                hi = home.right() - self.width()
                if hi > lo:
                    self._target_x = random.randint(lo, hi)
                    self._behaviour = WizardState.WALK

    def _flee(self, cursor: QPoint, home: QRect, y: int):
        """Teleport away from the cursor to a random spot on the opposite side."""
        lo = home.left()
        hi = home.right() - self.width()
        if hi <= lo:
            return
        mid = (lo + hi) // 2
        if cursor.x() < mid:
            new_x = random.randint(mid, hi)          # cursor on the left → flee right
            self._facing = 1
        else:
            new_x = random.randint(lo, mid)          # cursor on the right → flee left
            self._facing = -1
        self.move(new_x, y)
        self._behaviour = WizardState.IDLE
        self._dwell_until = time.monotonic() + random.uniform(DWELL_MIN_S, DWELL_MAX_S)
        self._set_state(WizardState.WALK)

    # ----- rendering --------------------------------------------------------
    def paintEvent(self, event):
        frames = self._frames.get(self._state) or []
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        if frames:
            pm = frames[self._frame % len(frames)]
            x = (self.width() - pm.width()) // 2
            y = (self.height() - pm.height()) // 2
            if self._facing < 0:
                pm = pm.transformed(_flip_transform(), Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(x, y, pm)
        else:
            self._paint_placeholder(painter)
        painter.end()
        # Inherit the edit-mode dashed border / highlight from the base class.
        super().paintEvent(event)

    def _paint_placeholder(self, painter: QPainter):
        """Draw a simple vector wizard so the pet works before real art lands."""
        bob = 2 if (self._frame % 2 == 0) else -2
        cx = self.width() // 2
        tint = _STATE_TINT.get(self._state, _STATE_TINT[WizardState.IDLE])
        base_y = self.height() - 12 + (bob if self._state == WizardState.WALK else 0)

        # Robe (triangle body).
        robe = QPolygon([
            QPoint(cx, base_y - 52),
            QPoint(cx - 26, base_y),
            QPoint(cx + 26, base_y),
        ])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(tint))
        painter.drawPolygon(robe)

        # Face.
        painter.setBrush(QBrush(QColor(245, 224, 200)))
        painter.drawEllipse(QPoint(cx, base_y - 50), 11, 11)

        # Hat (triangle), tipped toward facing direction.
        tip = QPoint(cx + self._facing * 6, base_y - 78)
        hat = QPolygon([
            tip,
            QPoint(cx - 16, base_y - 54),
            QPoint(cx + 16, base_y - 54),
        ])
        painter.setBrush(QBrush(tint.darker(130)))
        painter.drawPolygon(hat)
        # Hat star.
        painter.setBrush(QBrush(QColor(255, 220, 90)))
        painter.drawEllipse(QPoint(cx, base_y - 62), 2, 2)

        # Staff + glow when casting/reacting.
        staff_x = cx + self._facing * 24
        painter.setPen(QPen(QColor(150, 110, 70), 3))
        painter.drawLine(staff_x, base_y, staff_x, base_y - 46)
        if self._state in (WizardState.CAST, WizardState.REACT):
            glow = QColor(tint)
            glow.setAlpha(160 if self._state == WizardState.CAST else 220)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(glow))
            r = 7 + (self._frame % 3)
            painter.drawEllipse(QPoint(staff_x, base_y - 48), r, r)

        # Tiny eyes; X-eyes when sad.
        painter.setPen(QPen(QColor(40, 40, 40), 2))
        ex = cx + self._facing * 2
        if self._state == WizardState.SAD:
            painter.drawText(QRect(ex - 8, base_y - 56, 16, 12),
                             Qt.AlignmentFlag.AlignCenter, "x x")
        else:
            painter.drawPoint(ex - 3, base_y - 51)
            painter.drawPoint(ex + 4, base_y - 51)

    # ----- interaction: click opens chat, drag repositions ------------------
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press_global = event.globalPosition().toPoint()
            self._press_origin = self.pos()
            self._dragging_user = False
            event.accept()
        elif event.button() == Qt.MouseButton.RightButton:
            self._show_context_menu(event.globalPosition().toPoint())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            return super().mouseMoveEvent(event)
        delta = event.globalPosition().toPoint() - self._press_global
        if not self._dragging_user and delta.manhattanLength() > DRAG_THRESHOLD:
            self._dragging_user = True
        if self._dragging_user:
            self.move(self._press_origin + delta)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging_user:
                self._dragging_user = False
                self.save_geometry()
                self._snap_to_floor()
            elif self.app is not None and hasattr(self.app, "toggle_chat"):
                self.app.toggle_chat()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _show_context_menu(self, global_pos: QPoint):
        menu = QMenu()
        hide_action = menu.addAction(t("wizard.menu.hide"))
        settings_action = menu.addAction(t("wizard.menu.settings"))
        chosen = menu.exec(global_pos)
        if chosen == hide_action and self.app is not None and hasattr(self.app, "toggle_wizard"):
            self.app.toggle_wizard()
        elif chosen == settings_action and self.app is not None and hasattr(self.app, "_open_settings"):
            self.app._open_settings()


def _flip_transform():
    from PyQt6.QtGui import QTransform
    return QTransform().scale(-1, 1)
