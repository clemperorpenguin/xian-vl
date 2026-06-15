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

"""Familiar Mode — a floating pixel companion that walks the bottom of the screen.

The familiar is the "chat mode equivalent" surfaced as a living desktop pet: it
wanders along the screen floor, dodges the cursor, reacts when a translation is
running, and opens the Chat sidebar when clicked. When Familiar Mode is on, the
familiar's speech bubble *replaces* the standalone translation popup for the
default capture path — results appear next to the familiar instead.

Five switchable familiars ship as built-in vector art: **wizard, witch, cat,
owl, lemonfae**. They differ only in appearance and in how they travel up to
the top of the screen and back (their "transit style"):

* wizard / lemonfae — teleport up in a poof, then *float* gently back down.
* witch / owl       — fly: the witch rides her broom, the owl spreads its
                      wings, both up and down.
* cat               — scampers up the nearest screen edge and climbs back down.

It does *no* inference itself — every state change is driven by signals the app
already emits, and all animation runs off plain ``QTimer`` ticks on the GUI
thread, so it never touches the inference hot path.

Art is fully optional. Drop sprite frames into ``familiar/<species>/`` at the
workspace root named ``<state>_<n>.png`` (e.g. ``familiar/witch/idle_0.png``,
``walk_0.png``, ``cast_0.png``, ``react_0.png``, ``sad_0.png``) and they are
picked up automatically. The wizard also accepts the legacy flat
``familiar/<state>_<n>.png`` layout. Until real art is dropped in, a vector
placeholder is drawn per species so the whole system is testable today.
"""

import logging
import os
import random
import time
from dataclasses import dataclass
from enum import Enum, auto
from html import escape as html_escape

from PyQt6.QtWidgets import QWidget, QLabel, QMenu, QVBoxLayout, QApplication
from PyQt6.QtCore import Qt, QTimer, QPoint, QRect
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QPixmap, QPolygon, QCursor, QGuiApplication,
    QFont, QPainterPath,
)

from mage.ui.overlay_base import MageOverlayWindow
from mage.ui.theme import accent_hex
from mage.resources import get_resource_path
from mage.familiar_recipe import coerce_recipe, TEST_RECIPES, TEST_RECIPE_ORDER
from shared_types.state import t

logger = logging.getLogger(__name__)

# --- tunables ---------------------------------------------------------------
SPRITE_W = 128
SPRITE_H = 128
# The built-in vector placeholders are authored in a 96px "design" box; the
# painter is scaled to the real sprite size so they fill it cleanly at any size.
_ART_DESIGN = 96
FLOOR_MARGIN = 6          # px above the screen's bottom edge
PERCH_MARGIN = 10         # px below the screen's top edge while perched
WALK_STEP = 4             # px advanced per behaviour tick while walking
FLY_STEP = 14             # px per tick while flying/climbing up
FLOAT_STEP = 6            # px per tick while floating/flying back down (gentler)
ANIM_INTERVAL_MS = 140    # frame-swap cadence
TICK_INTERVAL_MS = 33     # behaviour / movement cadence (~30 fps)
AVOID_RADIUS = 150        # cursor proximity that triggers an escape upward
DRAG_THRESHOLD = 6        # px of movement before a press becomes a drag
CAST_HOLD_S = 2.0         # how long a "thinking" pose lingers between updates
BUBBLE_MS = 9000          # speech-bubble lifetime
ERROR_BUBBLE_MS = 5000    # shorter lifetime for error bubbles
# Hold the react/sad pose for the full bubble lifetime so the familiar stays
# planted under its speech bubble instead of wandering off mid-display.
REACT_HOLD_S = BUBBLE_MS / 1000.0
SAD_HOLD_S = ERROR_BUBBLE_MS / 1000.0
DWELL_MIN_S = 1.5         # idle pause range between walks
DWELL_MAX_S = 4.5
PERCH_MIN_S = 2.5         # how long the familiar lingers at the top
PERCH_MAX_S = 6.0
# The familiar mostly stays on the floor; ascending to the top is the exception.
IDLE_ASCEND_CHANCE = 0.08   # chance an idle decision becomes an ascent
CURSOR_ASCEND_CHANCE = 0.2  # chance a cursor-dodge escapes up instead of along the floor
TELEPORT_STEP = 0.14      # teleport blink progress per tick (0..1)
BUBBLE_MAX_W = 360        # speech-bubble width cap


class FamiliarState(Enum):
    IDLE = auto()
    WALK = auto()
    CAST = auto()    # a translation is running
    REACT = auto()   # a translation just finished
    SAD = auto()     # a translation errored


class TransitPhase(Enum):
    """Where the familiar is in its journey to/from the top of the screen."""
    GROUNDED = auto()    # normal floor wandering
    ASCENDING = auto()   # travelling up
    PERCHED = auto()     # sitting at the top
    DESCENDING = auto()  # travelling back down


class FamiliarSpecies(Enum):
    WIZARD = "wizard"
    WITCH = "witch"
    CAT = "cat"
    OWL = "owl"
    LEMONFAE = "lemonfae"
    CUSTOM = "custom"     # conjured: art comes from a recipe, not a fixed _Profile

    @classmethod
    def from_value(cls, value) -> "FamiliarSpecies":
        try:
            return cls(str(value))
        except ValueError:
            return cls.WIZARD


class TransitStyle(Enum):
    """How a species moves between the floor and its perch."""
    TELEPORT_FLOAT = auto()  # up: instant poof   down: gentle float
    FLIGHT = auto()          # up & down: airborne glide (broom / wings)
    CLIMB = auto()           # up & down: crawl along the nearest vertical edge


# ---------------------------------------------------------------------------
# Vector art — one renderer per species, used until real sprites are dropped in.
# Each renderer draws the current mood pose, plus a transit pose (broom/wings/
# climb cling) while the familiar is travelling.
# ---------------------------------------------------------------------------
class _FamiliarArt:
    """Base vector renderer. Subclasses override ``draw``."""

    def base_y(self, h: int, state: FamiliarState, frame: int) -> int:
        bob = 2 if (frame % 2 == 0) else -2
        return h - 12 + (bob if state == FamiliarState.WALK else 0)

    def draw(self, p: QPainter, state, frame, facing, transit, w, h):
        raise NotImplementedError

    # Shared helpers ------------------------------------------------------
    def _eyes(self, p: QPainter, cx, eye_y, facing, state, spread=4):
        p.setPen(QPen(QColor(40, 40, 40), 2))
        ex = cx + facing * 2
        if state == FamiliarState.SAD:
            p.drawText(QRect(ex - 8, eye_y - 6, 16, 12),
                       Qt.AlignmentFlag.AlignCenter, "x x")
        else:
            p.drawPoint(ex - spread // 2 - 1, eye_y)
            p.drawPoint(ex + spread // 2 + 1, eye_y)


class _WizardArt(_FamiliarArt):
    TINT = {
        FamiliarState.IDLE: QColor(80, 130, 220),
        FamiliarState.WALK: QColor(90, 160, 235),
        FamiliarState.CAST: QColor(150, 90, 230),
        FamiliarState.REACT: QColor(80, 200, 130),
        FamiliarState.SAD: QColor(130, 130, 140),
    }

    def draw(self, p, state, frame, facing, transit, w, h):
        cx = w // 2
        tint = self.TINT.get(state, self.TINT[FamiliarState.IDLE])
        base_y = self.base_y(h, state, frame)

        # Robe (triangle body).
        robe = QPolygon([
            QPoint(cx, base_y - 52),
            QPoint(cx - 26, base_y),
            QPoint(cx + 26, base_y),
        ])
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(tint))
        p.drawPolygon(robe)

        # Face.
        p.setBrush(QBrush(QColor(245, 224, 200)))
        p.drawEllipse(QPoint(cx, base_y - 50), 11, 11)

        # Hat (triangle), tipped toward facing direction.
        tip = QPoint(cx + facing * 6, base_y - 78)
        hat = QPolygon([tip, QPoint(cx - 16, base_y - 54), QPoint(cx + 16, base_y - 54)])
        p.setBrush(QBrush(tint.darker(130)))
        p.drawPolygon(hat)
        p.setBrush(QBrush(QColor(255, 220, 90)))
        p.drawEllipse(QPoint(cx, base_y - 62), 2, 2)

        # Staff + glow when casting/reacting.
        staff_x = cx + facing * 24
        p.setPen(QPen(QColor(150, 110, 70), 3))
        p.drawLine(staff_x, base_y, staff_x, base_y - 46)
        if state in (FamiliarState.CAST, FamiliarState.REACT):
            glow = QColor(tint)
            glow.setAlpha(160 if state == FamiliarState.CAST else 220)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(glow))
            r = 7 + (frame % 3)
            p.drawEllipse(QPoint(staff_x, base_y - 48), r, r)

        self._eyes(p, cx, base_y - 51, facing, state)


class _WitchArt(_FamiliarArt):
    ROBE = QColor(110, 60, 150)
    HAT = QColor(40, 30, 60)

    def draw(self, p, state, frame, facing, transit, w, h):
        cx = w // 2
        base_y = self.base_y(h, state, frame)
        flying = transit in (TransitPhase.ASCENDING, TransitPhase.PERCHED,
                             TransitPhase.DESCENDING)

        # Broomstick beneath her while flying.
        if flying:
            p.setPen(QPen(QColor(150, 110, 70), 4))
            p.drawLine(cx - 30, base_y + 2, cx + 30, base_y - 6)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(QColor(210, 170, 90)))
            bristles = QPolygon([
                QPoint(cx + 30, base_y - 14), QPoint(cx + 30, base_y + 2),
                QPoint(cx + 44, base_y - 6),
            ])
            p.drawPolygon(bristles)

        # Robe.
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self.ROBE if state != FamiliarState.SAD else QColor(120, 120, 128)))
        robe = QPolygon([
            QPoint(cx, base_y - 50), QPoint(cx - 24, base_y), QPoint(cx + 24, base_y),
        ])
        p.drawPolygon(robe)

        # Face (greenish witch skin).
        p.setBrush(QBrush(QColor(190, 220, 170)))
        p.drawEllipse(QPoint(cx, base_y - 48), 11, 11)

        # Wide-brim pointed hat.
        p.setBrush(QBrush(self.HAT))
        p.drawPolygon(QPolygon([
            QPoint(cx + facing * 8, base_y - 80),
            QPoint(cx - 14, base_y - 56), QPoint(cx + 14, base_y - 56),
        ]))
        p.drawRect(cx - 22, base_y - 58, 44, 4)  # brim

        # Casting glow at fingertip.
        if state in (FamiliarState.CAST, FamiliarState.REACT):
            glow = QColor(150, 230, 120) if state == FamiliarState.REACT else QColor(180, 120, 240)
            glow.setAlpha(200)
            p.setBrush(QBrush(glow))
            r = 6 + (frame % 3)
            p.drawEllipse(QPoint(cx + facing * 22, base_y - 30), r, r)

        self._eyes(p, cx, base_y - 49, facing, state)


class _CatArt(_FamiliarArt):
    FUR = QColor(70, 70, 80)

    def draw(self, p, state, frame, facing, transit, w, h):
        cx = w // 2
        base_y = self.base_y(h, state, frame)
        climbing = transit in (TransitPhase.ASCENDING, TransitPhase.PERCHED,
                               TransitPhase.DESCENDING)
        fur = self.FUR if state != FamiliarState.SAD else QColor(120, 120, 128)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(fur))

        if climbing:
            # Upright clinging pose, paws reaching up.
            body = QRect(cx - 14, base_y - 46, 28, 44)
            p.drawRoundedRect(body, 12, 12)
            # Paws up.
            p.drawEllipse(QPoint(cx - 12, base_y - 48), 5, 5)
            p.drawEllipse(QPoint(cx + 12, base_y - 48), 5, 5)
            head_c = QPoint(cx, base_y - 50)
            tail_start = QPoint(cx, base_y)
        else:
            # Crouched body.
            body = QRect(cx - 20, base_y - 28, 40, 28)
            p.drawRoundedRect(body, 12, 12)
            head_c = QPoint(cx + facing * 12, base_y - 30)
            tail_start = QPoint(cx - facing * 18, base_y - 18)

        # Head.
        p.drawEllipse(head_c, 13, 12)
        # Ears.
        p.drawPolygon(QPolygon([
            QPoint(head_c.x() - 11, head_c.y() - 6), QPoint(head_c.x() - 4, head_c.y() - 18),
            QPoint(head_c.x() - 1, head_c.y() - 7),
        ]))
        p.drawPolygon(QPolygon([
            QPoint(head_c.x() + 11, head_c.y() - 6), QPoint(head_c.x() + 4, head_c.y() - 18),
            QPoint(head_c.x() + 1, head_c.y() - 7),
        ]))
        # Tail (curls with frame).
        curl = 6 if frame % 2 == 0 else 12
        p.setPen(QPen(fur, 5))
        p.drawLine(tail_start.x(), tail_start.y(),
                   tail_start.x() - facing * 10, tail_start.y() - curl)

        # Eyes (glowing green) + sad fallback.
        if state == FamiliarState.SAD:
            self._eyes(p, head_c.x(), head_c.y(), facing, state)
        else:
            eye = QColor(150, 230, 120) if state in (FamiliarState.CAST, FamiliarState.REACT) else QColor(120, 200, 110)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(eye))
            p.drawEllipse(QPoint(head_c.x() - 5, head_c.y()), 2, 3)
            p.drawEllipse(QPoint(head_c.x() + 5, head_c.y()), 2, 3)


class _OwlArt(_FamiliarArt):
    BODY = QColor(150, 110, 70)

    def draw(self, p, state, frame, facing, transit, w, h):
        cx = w // 2
        base_y = self.base_y(h, state, frame)
        flying = transit in (TransitPhase.ASCENDING, TransitPhase.PERCHED,
                             TransitPhase.DESCENDING)
        body = self.BODY if state != FamiliarState.SAD else QColor(120, 120, 128)
        p.setPen(Qt.PenStyle.NoPen)

        # Wings — spread while flying (flap with frame), tucked otherwise.
        p.setBrush(QBrush(body.darker(130)))
        if flying:
            flap = -10 if frame % 2 == 0 else 4
            p.drawPolygon(QPolygon([
                QPoint(cx - 14, base_y - 34), QPoint(cx - 46, base_y - 34 + flap),
                QPoint(cx - 14, base_y - 14),
            ]))
            p.drawPolygon(QPolygon([
                QPoint(cx + 14, base_y - 34), QPoint(cx + 46, base_y - 34 + flap),
                QPoint(cx + 14, base_y - 14),
            ]))

        # Round body.
        p.setBrush(QBrush(body))
        p.drawEllipse(QPoint(cx, base_y - 24), 22, 26)
        # Belly.
        p.setBrush(QBrush(QColor(235, 215, 180)))
        p.drawEllipse(QPoint(cx, base_y - 18), 12, 16)

        # Ear tufts.
        p.setBrush(QBrush(body.darker(120)))
        p.drawPolygon(QPolygon([
            QPoint(cx - 16, base_y - 44), QPoint(cx - 9, base_y - 58), QPoint(cx - 5, base_y - 44),
        ]))
        p.drawPolygon(QPolygon([
            QPoint(cx + 16, base_y - 44), QPoint(cx + 9, base_y - 58), QPoint(cx + 5, base_y - 44),
        ]))

        # Big eyes.
        if state == FamiliarState.SAD:
            self._eyes(p, cx, base_y - 36, facing, state, spread=14)
        else:
            for sx in (-9, 9):
                p.setBrush(QBrush(QColor(245, 245, 245)))
                p.drawEllipse(QPoint(cx + sx, base_y - 36), 8, 8)
                lit = state in (FamiliarState.CAST, FamiliarState.REACT)
                p.setBrush(QBrush(QColor(255, 190, 60) if lit else QColor(40, 40, 50)))
                p.drawEllipse(QPoint(cx + sx + facing * 2, base_y - 36), 3, 3)
        # Beak.
        p.setBrush(QBrush(QColor(240, 170, 60)))
        p.drawPolygon(QPolygon([
            QPoint(cx - 4, base_y - 28), QPoint(cx + 4, base_y - 28), QPoint(cx, base_y - 21),
        ]))


class _LemonfaeArt(_FamiliarArt):
    BODY = QColor(240, 215, 70)

    def draw(self, p, state, frame, facing, transit, w, h):
        cx = w // 2
        base_y = self.base_y(h, state, frame)
        body = self.BODY if state != FamiliarState.SAD else QColor(150, 150, 120)
        p.setPen(Qt.PenStyle.NoPen)

        # Translucent fairy wings (flutter with frame).
        wing = QColor(200, 240, 200, 150)
        p.setBrush(QBrush(wing))
        spread = 18 if frame % 2 == 0 else 14
        p.drawEllipse(QPoint(cx - 16, base_y - 40), spread, 12)
        p.drawEllipse(QPoint(cx + 16, base_y - 40), spread, 12)

        # Lemon-shaped body.
        p.setBrush(QBrush(body))
        p.drawEllipse(QRect(cx - 16, base_y - 44, 32, 40))
        # Little leaf on top.
        p.setBrush(QBrush(QColor(120, 200, 110)))
        p.drawPolygon(QPolygon([
            QPoint(cx, base_y - 46), QPoint(cx + 8, base_y - 54), QPoint(cx + 2, base_y - 44),
        ]))

        # Glow aura when active.
        if state in (FamiliarState.CAST, FamiliarState.REACT):
            glow = QColor(255, 245, 150, 120)
            p.setBrush(QBrush(glow))
            r = 26 + (frame % 3) * 2
            p.drawEllipse(QPoint(cx, base_y - 24), r, r)
            p.setBrush(QBrush(body))
            p.drawEllipse(QRect(cx - 16, base_y - 44, 32, 40))

        # Face.
        if state == FamiliarState.SAD:
            self._eyes(p, cx, base_y - 26, facing, state)
        else:
            p.setPen(QPen(QColor(70, 60, 30), 2))
            ex = cx + facing * 2
            p.drawPoint(ex - 5, base_y - 26)
            p.drawPoint(ex + 5, base_y - 26)
            # Smile.
            path = QPainterPath()
            path.moveTo(cx - 5, base_y - 19)
            path.quadTo(cx, base_y - 14, cx + 5, base_y - 19)
            p.drawPath(path)


class _RecipeArt(_FamiliarArt):
    """Parametric vector renderer driven by a coerced art recipe (see
    ``mage.familiar_recipe``). Composes a familiar from fixed shape primitives so
    any valid recipe always renders, animating with the same conventions as the
    built-in species (walk-bob, cast/react aura, sad X-eyes, transit poses)."""

    def __init__(self, recipe: dict):
        self.recipe = recipe
        pal = recipe["palette"]
        self.primary = QColor(pal["primary"])
        self.secondary = QColor(pal["secondary"])
        self.accent = QColor(pal["accent"])
        self.skin = QColor(pal["skin"])
        self.glow = QColor(recipe["glow_color"])

    def draw(self, p, state, frame, facing, transit, w, h):
        r = self.recipe
        cx = w // 2
        base_y = self.base_y(h, state, frame)
        sad = state == FamiliarState.SAD
        flying = transit in (TransitPhase.ASCENDING, TransitPhase.PERCHED,
                             TransitPhase.DESCENDING)
        body_col = QColor(120, 120, 128) if sad else QColor(self.primary)
        feats = r["features"]
        p.setPen(Qt.PenStyle.NoPen)

        # 0. aura behind the body while casting / reacting.
        if state in (FamiliarState.CAST, FamiliarState.REACT) and not sad:
            self._aura(p, cx, base_y, frame, state)
        # 1. broom beneath (only when airborne).
        if flying and r["accessory"] == "broom":
            self._broom(p, cx, base_y)
        # 2-3. behind-body features.
        if "wings" in feats:
            self._wings(p, cx, base_y, frame, flying)
        if "tail" in feats:
            self._tail(p, cx, base_y, facing, frame, body_col)
        # 4. body + head.
        head_c, head_r = self._body(p, r["body"], r["head"], cx, base_y, facing, body_col)
        # 5. spikes along the back.
        if "spikes" in feats:
            self._spikes(p, cx, base_y, body_col)
        # 6-7. head extras.
        self._ears(p, r["ears"], head_c, head_r, body_col)
        if "halo" in feats:
            self._halo(p, head_c, head_r)
        if "antennae" in feats:
            self._antennae(p, head_c, head_r)
        self._headwear(p, r["headwear"], head_c, head_r, facing)
        # 8. face.
        if "whiskers" in feats:
            self._whiskers(p, head_c, head_r)
        self._eyes_for(p, r["eyes"], head_c, facing, sad, state)
        # 8b. face / accent features.
        if "glasses" in feats:
            self._glasses(p, head_c)
        if "fangs" in feats and not sad:
            self._fangs(p, head_c, head_r)
        if "gem" in feats:
            self._gem(p, head_c, head_r)
        if "scarf" in feats:
            self._scarf(p, head_c, head_r)
        if "flame" in feats:
            self._flame(p, head_c, head_r, frame)
        # 9. held accessory (not while flying — hands are busy).
        if r["accessory"] in ("staff", "wand", "orb", "lantern", "book") and not flying:
            self._held(p, r["accessory"], cx, base_y, facing, state, frame)

    # -- body ----------------------------------------------------------------
    def _body(self, p, shape, head_kind, cx, base_y, facing, col):
        p.setBrush(QBrush(col))
        if shape == "triangle":
            p.drawPolygon(QPolygon([
                QPoint(cx, base_y - 50), QPoint(cx - 24, base_y), QPoint(cx + 24, base_y)]))
            head_c, head_r, merge_y = QPoint(cx, base_y - 48), 11, base_y - 40
        elif shape == "tall":
            p.drawRoundedRect(QRect(cx - 15, base_y - 54, 30, 54), 12, 12)
            head_c, head_r, merge_y = QPoint(cx, base_y - 56), 12, base_y - 38
        elif shape == "blob":
            p.drawRoundedRect(QRect(cx - 22, base_y - 40, 44, 40), 18, 18)
            head_c, head_r, merge_y = QPoint(cx, base_y - 40), 13, base_y - 28
        elif shape == "egg":
            p.drawEllipse(QRect(cx - 18, base_y - 52, 36, 52))
            head_c, head_r, merge_y = QPoint(cx, base_y - 42), 11, base_y - 32
        elif shape == "ghost":
            self._ghost_body(p, cx, base_y)
            head_c, head_r, merge_y = QPoint(cx, base_y - 38), 11, base_y - 34
        elif shape == "quad":
            for lx in (-16, -6, 6, 16):
                p.drawRect(cx + lx - 2, base_y - 8, 5, 8)
            p.drawRoundedRect(QRect(cx - 22, base_y - 26, 44, 20), 10, 10)
            head_c, head_r = QPoint(cx + facing * 14, base_y - 30), 11
            p.setBrush(QBrush(self.skin))
            p.drawEllipse(head_c, head_r, head_r)
            return head_c, head_r          # quad always has a separate head
        else:  # round
            p.drawEllipse(QPoint(cx, base_y - 20), 20, 22)
            head_c, head_r, merge_y = QPoint(cx, base_y - 44), 12, base_y - 32

        if head_kind == "merged":
            merged = QPoint(cx, merge_y)
            p.setBrush(QBrush(self.skin))
            p.drawEllipse(merged, 10, 9)
            return merged, 10
        p.setBrush(QBrush(self.skin))
        p.drawEllipse(head_c, head_r, head_r)
        return head_c, head_r

    def _ghost_body(self, p, cx, base_y):
        top = base_y - 50
        path = QPainterPath()
        path.moveTo(cx - 20, base_y - 6)
        path.lineTo(cx - 20, top + 18)
        path.quadTo(cx - 20, top, cx, top)
        path.quadTo(cx + 20, top, cx + 20, top + 18)
        path.lineTo(cx + 20, base_y - 6)
        # Wavy hem.
        path.quadTo(cx + 13, base_y + 2, cx + 7, base_y - 6)
        path.quadTo(cx, base_y + 2, cx - 7, base_y - 6)
        path.quadTo(cx - 13, base_y + 2, cx - 20, base_y - 6)
        path.closeSubpath()
        p.drawPath(path)

    # -- behind-body features ------------------------------------------------
    def _wings(self, p, cx, base_y, frame, flying):
        wcol = QColor(self.secondary)
        wcol.setAlpha(235)
        p.setBrush(QBrush(wcol))
        flap = (-8 if frame % 2 == 0 else 2) if flying else 0
        spread = 30 if flying else 18
        p.drawPolygon(QPolygon([
            QPoint(cx - 12, base_y - 34), QPoint(cx - 12 - spread, base_y - 40 + flap),
            QPoint(cx - 10, base_y - 14)]))
        p.drawPolygon(QPolygon([
            QPoint(cx + 12, base_y - 34), QPoint(cx + 12 + spread, base_y - 40 + flap),
            QPoint(cx + 10, base_y - 14)]))

    def _tail(self, p, cx, base_y, facing, frame, col):
        curl = 6 if frame % 2 == 0 else 12
        p.setPen(QPen(col, 5))
        p.drawLine(cx - facing * 18, base_y - 12, cx - facing * 30, base_y - 12 - curl)
        p.setPen(Qt.PenStyle.NoPen)

    def _spikes(self, p, cx, base_y, col):
        p.setBrush(QBrush(self.secondary))
        for dx in (-12, 0, 12):
            p.drawPolygon(QPolygon([
                QPoint(cx + dx - 4, base_y - 40), QPoint(cx + dx, base_y - 52),
                QPoint(cx + dx + 4, base_y - 40)]))

    # -- head extras ---------------------------------------------------------
    def _ears(self, p, ears, head_c, head_r, col):
        if ears == "none":
            return
        hx, hy = head_c.x(), head_c.y()
        top = hy - head_r
        if ears == "long":
            p.setBrush(QBrush(col))
            p.drawEllipse(QPoint(hx - 5, top - 8), 3, 12)
            p.drawEllipse(QPoint(hx + 5, top - 8), 3, 12)
            return
        if ears == "round":
            p.setBrush(QBrush(col))
            p.drawEllipse(QPoint(hx - head_r + 2, top + 2), 5, 5)
            p.drawEllipse(QPoint(hx + head_r - 2, top + 2), 5, 5)
            return
        if ears == "floppy":
            p.setBrush(QBrush(col))
            p.drawEllipse(QPoint(hx - head_r, hy + 1), 4, 9)
            p.drawEllipse(QPoint(hx + head_r, hy + 1), 4, 9)
            return
        if ears == "horns":
            p.setBrush(QBrush(self.accent))
            p.drawPolygon(QPolygon([
                QPoint(hx - head_r + 2, top + 4), QPoint(hx - head_r - 6, top - 8),
                QPoint(hx - 3, top + 2)]))
            p.drawPolygon(QPolygon([
                QPoint(hx + head_r - 2, top + 4), QPoint(hx + head_r + 6, top - 8),
                QPoint(hx + 3, top + 2)]))
            return
        # pointed / tufts (tufts are a touch smaller and more upright).
        p.setBrush(QBrush(col))
        h = 12 if ears == "pointed" else 8
        p.drawPolygon(QPolygon([
            QPoint(hx - head_r + 2, top + 4), QPoint(hx - 5, top - h), QPoint(hx + 1, top + 3)]))
        p.drawPolygon(QPolygon([
            QPoint(hx + head_r - 2, top + 4), QPoint(hx + 5, top - h), QPoint(hx - 1, top + 3)]))

    def _halo(self, p, head_c, head_r):
        p.setPen(QPen(self.accent, 3))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPoint(head_c.x(), head_c.y() - head_r - 9), 11, 4)
        p.setPen(Qt.PenStyle.NoPen)

    def _antennae(self, p, head_c, head_r):
        hx, top = head_c.x(), head_c.y() - head_r
        p.setPen(QPen(self.secondary, 2))
        p.drawLine(hx - 5, top, hx - 9, top - 10)
        p.drawLine(hx + 5, top, hx + 9, top - 10)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(self.accent))
        p.drawEllipse(QPoint(hx - 9, top - 11), 2, 2)
        p.drawEllipse(QPoint(hx + 9, top - 11), 2, 2)

    def _headwear(self, p, hw, head_c, head_r, facing):
        if hw == "none":
            return
        hx, hy = head_c.x(), head_c.y()
        top = hy - head_r
        if hw == "pointed_hat":
            p.setBrush(QBrush(self.secondary))
            p.drawPolygon(QPolygon([
                QPoint(hx + facing * 5, top - 22), QPoint(hx - 14, top + 2), QPoint(hx + 14, top + 2)]))
            p.drawRect(hx - 18, top, 36, 3)
            p.setBrush(QBrush(self.accent))
            p.drawEllipse(QPoint(hx, top - 8), 2, 2)
        elif hw == "crown":
            p.setBrush(QBrush(self.accent))
            p.drawPolygon(QPolygon([
                QPoint(hx - 12, top + 3), QPoint(hx - 12, top - 4), QPoint(hx - 6, top + 2),
                QPoint(hx, top - 6), QPoint(hx + 6, top + 2), QPoint(hx + 12, top - 4),
                QPoint(hx + 12, top + 3)]))
        elif hw == "hood":
            p.setBrush(QBrush(self.secondary))
            p.drawChord(QRect(hx - head_r - 1, top - 3, 2 * (head_r + 1), 2 * (head_r + 1)),
                        0, 180 * 16)
        elif hw == "leaf":
            p.setBrush(QBrush(QColor(120, 200, 110)))
            p.drawPolygon(QPolygon([
                QPoint(hx, top), QPoint(hx + 9, top - 11), QPoint(hx + 2, top + 1)]))
        elif hw == "bow":
            by = top - 1
            p.setBrush(QBrush(self.accent))
            p.drawPolygon(QPolygon([
                QPoint(hx, by), QPoint(hx - 9, by - 5), QPoint(hx - 9, by + 5)]))
            p.drawPolygon(QPolygon([
                QPoint(hx, by), QPoint(hx + 9, by - 5), QPoint(hx + 9, by + 5)]))
            p.drawEllipse(QPoint(hx, by), 2, 2)
        elif hw == "antlers":
            p.setPen(QPen(QColor(155, 118, 83), 2))
            for s in (-1, 1):
                bx = hx + s * 5
                p.drawLine(bx, top, bx + s * 4, top - 12)
                p.drawLine(bx + s * 2, top - 6, bx + s * 8, top - 8)
                p.drawLine(bx + s * 4, top - 12, bx + s * 9, top - 14)
            p.setPen(Qt.PenStyle.NoPen)

    def _whiskers(self, p, head_c, head_r):
        hx, hy = head_c.x(), head_c.y()
        p.setPen(QPen(QColor(235, 235, 235), 1))
        for dy in (-1, 2, 5):
            p.drawLine(hx - head_r, hy + dy, hx - head_r - 9, hy + dy - 2)
            p.drawLine(hx + head_r, hy + dy, hx + head_r + 9, hy + dy - 2)
        p.setPen(Qt.PenStyle.NoPen)

    def _eyes_for(self, p, eyes, head_c, facing, sad, state):
        hx, hy = head_c.x(), head_c.y()
        if sad:
            p.setPen(QPen(QColor(40, 40, 40), 2))
            p.drawText(QRect(hx - 8, hy - 6, 16, 12), Qt.AlignmentFlag.AlignCenter, "x x")
            p.setPen(Qt.PenStyle.NoPen)
            return
        lit = state in (FamiliarState.CAST, FamiliarState.REACT)
        if eyes == "big":
            for sx in (-6, 6):
                p.setBrush(QBrush(QColor(245, 245, 245)))
                p.drawEllipse(QPoint(hx + sx, hy), 5, 5)
                p.setBrush(QBrush(self.glow if lit else QColor(40, 40, 50)))
                p.drawEllipse(QPoint(hx + sx + facing, hy), 2, 2)
        elif eyes == "sleepy":
            p.setPen(QPen(QColor(40, 40, 40), 2))
            p.drawLine(hx - 7, hy, hx - 2, hy)
            p.drawLine(hx + 2, hy, hx + 7, hy)
            p.setPen(Qt.PenStyle.NoPen)
        elif eyes == "wink":
            p.setPen(QPen(QColor(40, 40, 40), 2))
            p.drawPoint(hx - 4, hy)
            p.drawLine(hx + 2, hy, hx + 7, hy)   # winking eye
            p.setPen(Qt.PenStyle.NoPen)
        elif eyes == "glow":
            p.setBrush(QBrush(self.glow if lit else self.accent))
            p.drawEllipse(QPoint(hx - 5, hy), 3, 3)
            p.drawEllipse(QPoint(hx + 5, hy), 3, 3)
        else:  # dots
            p.setPen(QPen(QColor(40, 40, 40), 2))
            p.drawPoint(hx - 4, hy)
            p.drawPoint(hx + 4, hy)
            p.setPen(Qt.PenStyle.NoPen)

    # -- face / accent features ---------------------------------------------
    def _glasses(self, p, head_c):
        hx, hy = head_c.x(), head_c.y()
        p.setPen(QPen(QColor(40, 40, 40), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawEllipse(QPoint(hx - 5, hy), 4, 4)
        p.drawEllipse(QPoint(hx + 5, hy), 4, 4)
        p.drawLine(hx - 1, hy, hx + 1, hy)
        p.setPen(Qt.PenStyle.NoPen)

    def _fangs(self, p, head_c, head_r):
        hx, my = head_c.x(), head_c.y() + head_r - 3
        p.setBrush(QBrush(QColor(250, 250, 250)))
        p.drawPolygon(QPolygon([QPoint(hx - 5, my), QPoint(hx - 3, my), QPoint(hx - 4, my + 4)]))
        p.drawPolygon(QPolygon([QPoint(hx + 5, my), QPoint(hx + 3, my), QPoint(hx + 4, my + 4)]))

    def _gem(self, p, head_c, head_r):
        hx, ty = head_c.x(), head_c.y() - head_r + 3
        p.setBrush(QBrush(self.accent))
        p.drawPolygon(QPolygon([
            QPoint(hx, ty - 3), QPoint(hx + 3, ty), QPoint(hx, ty + 3), QPoint(hx - 3, ty)]))

    def _scarf(self, p, head_c, head_r):
        hx, ny = head_c.x(), head_c.y() + head_r - 2
        p.setBrush(QBrush(self.accent))
        p.drawRect(hx - 10, ny, 20, 4)
        p.drawPolygon(QPolygon([
            QPoint(hx + 6, ny + 4), QPoint(hx + 12, ny + 11), QPoint(hx + 6, ny + 11)]))

    def _flame(self, p, head_c, head_r, frame):
        hx, top = head_c.x(), head_c.y() - head_r
        h = 10 if frame % 2 == 0 else 14
        p.setBrush(QBrush(QColor(255, 122, 24)))
        p.drawPolygon(QPolygon([QPoint(hx - 5, top + 2), QPoint(hx, top - h), QPoint(hx + 5, top + 2)]))
        p.setBrush(QBrush(self.accent))
        p.drawPolygon(QPolygon([QPoint(hx - 2, top), QPoint(hx, top - h + 5), QPoint(hx + 2, top)]))

    # -- accessories & aura --------------------------------------------------
    def _broom(self, p, cx, base_y):
        p.setPen(QPen(QColor(150, 110, 70), 4))
        p.drawLine(cx - 30, base_y + 2, cx + 30, base_y - 6)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor(210, 170, 90)))
        p.drawPolygon(QPolygon([
            QPoint(cx + 30, base_y - 14), QPoint(cx + 30, base_y + 2), QPoint(cx + 44, base_y - 6)]))

    def _held(self, p, acc, cx, base_y, facing, state, frame):
        sx = cx + facing * 22
        lit = state in (FamiliarState.CAST, FamiliarState.REACT)
        if acc == "staff":
            p.setPen(QPen(QColor(150, 110, 70), 3))
            p.drawLine(sx, base_y, sx, base_y - 44)
            p.setPen(Qt.PenStyle.NoPen)
            if lit:
                g = QColor(self.glow)
                g.setAlpha(220)
                p.setBrush(QBrush(g))
                p.drawEllipse(QPoint(sx, base_y - 46), 6 + (frame % 3), 6 + (frame % 3))
        elif acc == "wand":
            p.setPen(QPen(QColor(120, 90, 60), 2))
            p.drawLine(sx, base_y - 10, sx + facing * 8, base_y - 26)
            p.setPen(Qt.PenStyle.NoPen)
            p.setBrush(QBrush(self.accent))
            p.drawEllipse(QPoint(sx + facing * 8, base_y - 28), 3, 3)
        elif acc == "lantern":
            p.setPen(QPen(QColor(120, 90, 60), 2))
            p.drawLine(sx, base_y - 20, sx, base_y - 30)
            p.setPen(Qt.PenStyle.NoPen)
            g = QColor(self.glow)
            g.setAlpha(220)
            p.setBrush(QBrush(g))
            p.drawRect(sx - 4, base_y - 20, 8, 10)
        elif acc == "book":
            p.setBrush(QBrush(self.secondary))
            p.drawRect(sx - 6, base_y - 22, 12, 9)
            p.setBrush(QBrush(self.skin))
            p.drawRect(sx - 4, base_y - 21, 8, 7)
        else:  # orb
            g = QColor(self.glow)
            g.setAlpha(210)
            p.setBrush(QBrush(g))
            p.drawEllipse(QPoint(sx, base_y - 30), 7, 7)

    def _aura(self, p, cx, base_y, frame, state):
        g = QColor(self.glow)
        g.setAlpha(150 if state == FamiliarState.CAST else 200)
        p.setBrush(QBrush(g))
        r = 24 + (frame % 3) * 2
        p.drawEllipse(QPoint(cx, base_y - 22), r, r)


@dataclass
class _Profile:
    species: FamiliarSpecies
    transit: TransitStyle
    art: _FamiliarArt
    glow: QColor            # teleport-sparkle / accent tint


_PROFILES: dict[FamiliarSpecies, _Profile] = {
    FamiliarSpecies.WIZARD: _Profile(
        FamiliarSpecies.WIZARD, TransitStyle.TELEPORT_FLOAT, _WizardArt(), QColor(150, 90, 230)),
    FamiliarSpecies.WITCH: _Profile(
        FamiliarSpecies.WITCH, TransitStyle.FLIGHT, _WitchArt(), QColor(150, 230, 120)),
    FamiliarSpecies.CAT: _Profile(
        FamiliarSpecies.CAT, TransitStyle.CLIMB, _CatArt(), QColor(120, 200, 110)),
    FamiliarSpecies.OWL: _Profile(
        FamiliarSpecies.OWL, TransitStyle.FLIGHT, _OwlArt(), QColor(255, 190, 60)),
    FamiliarSpecies.LEMONFAE: _Profile(
        FamiliarSpecies.LEMONFAE, TransitStyle.TELEPORT_FLOAT, _LemonfaeArt(), QColor(255, 235, 120)),
}

# Recipe "transit" vocabulary → the engine's transit styles. ("teleport" always
# blinks up and floats down; the recipe's float_down flag is reserved for a
# future teleport-both-ways variant.)
_RECIPE_TRANSIT = {
    "teleport": TransitStyle.TELEPORT_FLOAT,
    "fly": TransitStyle.FLIGHT,
    "climb": TransitStyle.CLIMB,
}


def _build_custom_profile(recipe: dict) -> _Profile:
    """Construct an on-the-fly profile for a conjured (recipe-driven) familiar."""
    style = _RECIPE_TRANSIT.get(recipe.get("transit"), TransitStyle.TELEPORT_FLOAT)
    return _Profile(FamiliarSpecies.CUSTOM, style, _RecipeArt(recipe),
                    QColor(recipe["glow_color"]))


class FamiliarSpeechBubble(MageOverlayWindow):
    """Auto-closing speech bubble shown above the familiar.

    In Familiar Mode this is the primary translation display, so it shows the
    original text (dim) above the translation (bright) much like the normal
    result popup, wraps long content, and auto-fades.
    """

    def __init__(self, app=None, parent=None):
        super().__init__(window_id="familiar_bubble", app=app, parent=parent)
        # A speech bubble should never be draggable or steal interaction.
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        text_size = 14
        if app is not None and hasattr(app, "settings") and app.settings:
            try:
                text_size = int(app.settings.value("overlay_text_size", 14))
            except (TypeError, ValueError):
                text_size = 14

        self._label = QLabel(self)
        self._label.setWordWrap(True)
        self._label.setTextFormat(Qt.TextFormat.RichText)
        self._label.setMaximumWidth(BUBBLE_MAX_W)
        self._label.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self._label.setStyleSheet(
            f"QLabel {{ color: #f4f4f4; background: rgba(18,18,22,236);"
            f" border: 1px solid {accent_hex()}; border-radius: 12px;"
            f" padding: 10px 13px; font-size: {text_size}px; }}"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._label)

        self._close_timer = QTimer(self)
        self._close_timer.setSingleShot(True)
        self._close_timer.timeout.connect(self.hide)

    def say(self, translation: str, original: str = "", anchor: QRect = None,
            ms: int = BUBBLE_MS):
        translation = (translation or "").strip()
        original = (original or "").strip()
        if not translation and not original:
            return

        parts = []
        if original:
            parts.append(
                f"<div style='color:#9aa0aa; font-size:0.85em; margin-bottom:4px;'>"
                f"{html_escape(original)}</div>"
            )
        if translation:
            parts.append(f"<div>{html_escape(translation)}</div>")
        self._label.setText("".join(parts))
        self._label.adjustSize()
        self.adjustSize()

        if anchor is None:
            anchor = QRect(self.x(), self.y(), SPRITE_W, SPRITE_H)
        # Anchor centred above the familiar, clamped onto its screen.
        x = anchor.center().x() - self.width() // 2
        y = anchor.top() - self.height() - 8
        screen = (QGuiApplication.screenAt(anchor.center())
                  or QGuiApplication.primaryScreen())
        geo = screen.availableGeometry()
        x = max(geo.left() + 4, min(x, geo.right() - self.width() - 4))
        if y < geo.top() + 4:
            y = anchor.bottom() + 8
        self.move(x, y)
        self.show()
        self.raise_()
        self._close_timer.start(ms)


class FamiliarPet(MageOverlayWindow):
    """A frameless, always-on-top sprite that lives on the desktop floor."""

    def __init__(self, app=None, parent=None, species=None):
        super().__init__(window_id="familiar_pet", app=app, parent=parent)
        self.setFixedSize(SPRITE_W, SPRITE_H)

        # Conjured familiars are driven by an art recipe rather than a fixed
        # profile; load it up front so a "custom" species can resolve.
        self._recipe = self._load_recipe()
        if species is None:
            species = self._species_from_settings()
        self._profile = self._profile_for(FamiliarSpecies.from_value(species))

        # --- animation / behaviour state ---
        self._state = FamiliarState.IDLE
        self._behaviour = FamiliarState.IDLE      # autonomous mode: IDLE or WALK
        self._facing = 1                          # +1 faces right, -1 faces left
        self._frame = 0
        self._target_x = self.x()
        now = time.monotonic()
        self._dwell_until = now + random.uniform(DWELL_MIN_S, DWELL_MAX_S)
        self._emote_until = 0.0                   # while > now, an emote overrides autonomy
        self._fleeing = False                     # mid cursor-dodge (decide once, not per tick)

        # --- vertical-transit state machine ---
        self._transit = TransitPhase.GROUNDED
        self._perch_until = 0.0
        self._teleport_p = 0.0                    # 0..1 blink progress (teleport style)
        self._climb_x = self.x()                  # target edge x while climbing

        # --- drag vs click discrimination ---
        self._press_global = QPoint()
        self._press_origin = QPoint()
        self._dragging_user = False

        # --- frames: real art if present, else vector placeholder ---
        self._frames: dict[FamiliarState, list[QPixmap]] = {}
        self._transit_frames: dict[TransitPhase, list[QPixmap]] = {}
        self._load_all_frames()

        self._bubble = FamiliarSpeechBubble(app=self.app, parent=self)

        self._snap_to_floor()

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_anim_tick)
        self._tick_timer = QTimer(self)
        self._tick_timer.timeout.connect(self._on_behaviour_tick)

    # ----- species ----------------------------------------------------------
    def _species_from_settings(self) -> FamiliarSpecies:
        if hasattr(self.app, "settings") and self.app.settings:
            try:
                from mage.settings_keys import KEY_FAMILIAR_TYPE
                return FamiliarSpecies.from_value(
                    self.app.settings.value(KEY_FAMILIAR_TYPE, "wizard"))
            except Exception:
                pass
        return FamiliarSpecies.WIZARD

    @property
    def species(self) -> FamiliarSpecies:
        return self._profile.species

    @property
    def current_recipe(self) -> dict:
        return self._recipe

    def _profile_for(self, species: FamiliarSpecies) -> _Profile:
        if species == FamiliarSpecies.CUSTOM:
            return _build_custom_profile(self._recipe)
        return _PROFILES[species]

    def set_species(self, species) -> None:
        """Swap the familiar's appearance in place, keeping its position/state."""
        target = FamiliarSpecies.from_value(
            species.value if isinstance(species, FamiliarSpecies) else species)
        if target == self._profile.species:
            return
        self._profile = self._profile_for(target)
        self._load_all_frames()
        self.update()
        logger.info("Familiar species set to %s", target.value)

    def _load_recipe(self) -> dict:
        """Load the saved conjure recipe, falling back to a bundled test recipe
        so the custom slot always renders something."""
        raw = None
        if hasattr(self.app, "settings") and self.app.settings:
            try:
                from mage.settings_keys import KEY_FAMILIAR_CUSTOM_RECIPE
                val = self.app.settings.value(KEY_FAMILIAR_CUSTOM_RECIPE)
                if val:
                    import json
                    raw = json.loads(val) if isinstance(val, str) else val
            except Exception:
                raw = None
        if not raw:
            raw = TEST_RECIPES[TEST_RECIPE_ORDER[0]]
        return coerce_recipe(raw)

    def set_recipe(self, recipe, persist: bool = True) -> None:
        """Apply a new conjure recipe; rebuilds the custom art if it's active."""
        self._recipe = coerce_recipe(recipe)
        if self._profile.species == FamiliarSpecies.CUSTOM:
            self._profile = _build_custom_profile(self._recipe)
            self._load_all_frames()
            self.update()
        if persist:
            self._persist_recipe()

    def _persist_recipe(self) -> None:
        import json
        data = json.dumps(self._recipe)
        if hasattr(self.app, "settings") and self.app.settings:
            try:
                from mage.settings_keys import KEY_FAMILIAR_CUSTOM_RECIPE
                self.app.settings.setValue(KEY_FAMILIAR_CUSTOM_RECIPE, data)
            except Exception:
                pass
        try:
            d = get_resource_path("familiar/custom")
            if os.path.isdir(d):
                with open(os.path.join(d, "recipe.json"), "w", encoding="utf-8") as f:
                    f.write(data)
        except Exception as e:
            logger.debug("Could not write recipe.json: %s", e)

    # ----- asset loading ----------------------------------------------------
    def _load_all_frames(self):
        for st in FamiliarState:
            self._frames[st] = self._load_frames(st.name.lower())
        # Optional dedicated travel poses (broom/wings/cling). Each falls back
        # to a mood state at paint time when absent, so they're never required.
        self._transit_frames = {
            TransitPhase.ASCENDING: self._load_frames("ascend"),
            TransitPhase.PERCHED: self._load_frames("perch"),
            TransitPhase.DESCENDING: self._load_frames("descend"),
        }

    def _load_frames(self, state_name: str) -> list[QPixmap]:
        species_name = self._profile.species.value
        frames: list[QPixmap] = []
        i = 0
        while True:
            # Prefer species-namespaced art; fall back to the legacy flat layout
            # for the wizard so existing drop-in folders keep working.
            candidates = [f"familiar/{species_name}/{state_name}_{i}.png"]
            if species_name == "wizard":
                candidates.append(f"familiar/{state_name}_{i}.png")
            path = None
            for rel in candidates:
                p = get_resource_path(rel)
                if p != rel and os.path.exists(p):
                    path = p
                    break
            if path is None:
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
        self._return_home_now()
        self._emote_until = time.monotonic() + CAST_HOLD_S
        self._set_state(FamiliarState.CAST)

    def on_result(self, text: str, original: str = "", with_bubble: bool = True):
        """A translation finished; react, and optionally pop a speech bubble."""
        self._return_home_now()
        self._emote_until = time.monotonic() + REACT_HOLD_S
        self._set_state(FamiliarState.REACT)
        if with_bubble and ((text and text.strip()) or (original and original.strip())):
            self._bubble.say(text, original=original, anchor=self.geometry())

    def on_error(self, msg: str = "", with_bubble: bool = True):
        self._return_home_now()
        self._emote_until = time.monotonic() + SAD_HOLD_S
        self._set_state(FamiliarState.SAD)
        if with_bubble:
            self._bubble.say(f"⚠ {msg}" if msg else "?!", anchor=self.geometry(), ms=ERROR_BUBBLE_MS)

    def shutdown(self):
        """Stop all timers and dismiss the bubble (called when disabling)."""
        self._anim_timer.stop()
        self._tick_timer.stop()
        if self._bubble:
            self._bubble.close()

    # The familiar must always be clickable, so block the click-through plumbing
    # the binder may apply to ordinary overlays.
    def set_click_through(self, click_through: bool):
        self._click_through = False
        if not self._edit_mode_active:
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)

    # ----- lifecycle --------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        self._transit = TransitPhase.GROUNDED
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
    def _set_state(self, state: FamiliarState):
        if state != self._state:
            self._state = state
            self._frame = 0
            self.update()

    def _active_frames(self) -> list[QPixmap]:
        """Pixmaps for the current moment: dedicated travel poses while in
        transit (ascend/perch/descend), else the current mood state. Travel
        poses fall back to walk/idle frames, and ultimately to the empty list
        that makes ``paintEvent`` draw the vector placeholder instead.
        """
        if self._transit == TransitPhase.ASCENDING:
            return self._transit_frames.get(TransitPhase.ASCENDING) or self._frames.get(FamiliarState.WALK) or []
        if self._transit == TransitPhase.PERCHED:
            return self._transit_frames.get(TransitPhase.PERCHED) or self._frames.get(FamiliarState.IDLE) or []
        if self._transit == TransitPhase.DESCENDING:
            return self._transit_frames.get(TransitPhase.DESCENDING) or self._frames.get(FamiliarState.WALK) or []
        return self._frames.get(self._state) or []

    def _on_anim_tick(self):
        frames = self._active_frames()
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

    def _perch_y(self, home: QRect) -> int:
        return home.top() + PERCH_MARGIN

    def _snap_to_floor(self):
        home = self._home_geometry()
        y = self._floor_y(home)
        x = max(home.left(), min(self.x(), home.right() - self.width()))
        self.move(x, y)

    def _return_home_now(self):
        """Abort any vertical transit and drop straight back to the floor.

        Used when a translation event arrives so the speech bubble still anchors
        next to the familiar at its usual floor position.
        """
        if self._transit != TransitPhase.GROUNDED:
            self._transit = TransitPhase.GROUNDED
            self._teleport_p = 0.0
            self._snap_to_floor()

    def _on_behaviour_tick(self):
        if self._dragging_user:
            return
        now = time.monotonic()

        # Vertical transit fully owns movement while it's running.
        if self._transit != TransitPhase.GROUNDED:
            self._tick_transit(now)
            return

        # An active emote (cast/react/sad) freezes wandering until it expires.
        if now < self._emote_until:
            return
        if self._state in (FamiliarState.CAST, FamiliarState.REACT, FamiliarState.SAD):
            self._behaviour = FamiliarState.IDLE
            self._dwell_until = now + random.uniform(DWELL_MIN_S, DWELL_MAX_S)
            self._set_state(FamiliarState.IDLE)

        home = self._home_geometry()
        y = self._floor_y(home)

        # Dodge the cursor. Most of the time the familiar just scurries away
        # along the floor; only occasionally does it escape up to the top.
        cursor = QCursor.pos()
        center = QPoint(self.x() + self.width() // 2, self.y() + self.height() // 2)
        near = (abs(cursor.x() - center.x()) < AVOID_RADIUS
                and abs(cursor.y() - center.y()) < AVOID_RADIUS)
        if near:
            if not self._fleeing:
                self._fleeing = True
                if random.random() < CURSOR_ASCEND_CHANCE:
                    self._begin_ascend(home, cursor)
                    return
                self._start_ground_flee(cursor, home)
            # else: already fleeing along the floor — fall through to WALK.
        else:
            self._fleeing = False

        if self._behaviour == FamiliarState.WALK:
            cur_x = self.x()
            dx = self._target_x - cur_x
            if abs(dx) <= WALK_STEP:
                self.move(self._target_x, y)
                self._behaviour = FamiliarState.IDLE
                self._dwell_until = now + random.uniform(DWELL_MIN_S, DWELL_MAX_S)
                self._set_state(FamiliarState.IDLE)
            else:
                self._facing = 1 if dx > 0 else -1
                self.move(cur_x + (WALK_STEP if dx > 0 else -WALK_STEP), y)
                self._set_state(FamiliarState.WALK)
        else:  # IDLE
            self.move(self.x(), y)  # keep glued to the floor if the screen changed
            self._set_state(FamiliarState.IDLE)
            if now >= self._dwell_until:
                # Occasionally wander *up* to perch instead of along the floor.
                if random.random() < IDLE_ASCEND_CHANCE:
                    self._begin_ascend(home, None)
                    return
                lo = home.left()
                hi = home.right() - self.width()
                if hi > lo:
                    self._target_x = random.randint(lo, hi)
                    self._behaviour = FamiliarState.WALK

    def _start_ground_flee(self, cursor: QPoint, home: QRect):
        """Scurry away from the cursor along the floor (no ascent)."""
        lo = home.left()
        hi = home.right() - self.width()
        if hi <= lo:
            return
        if cursor.x() < self.x() + self.width() // 2:
            self._target_x, self._facing = hi, 1     # cursor on the left → go right
        else:
            self._target_x, self._facing = lo, -1
        self._behaviour = FamiliarState.WALK
        self._set_state(FamiliarState.WALK)

    def _begin_ascend(self, home: QRect, cursor):
        """Start a trip to the top of the screen using the species' transit style."""
        self._transit = TransitPhase.ASCENDING
        self._teleport_p = 0.0
        self._behaviour = FamiliarState.IDLE
        self._set_state(FamiliarState.WALK)
        if self._profile.transit == TransitStyle.CLIMB:
            # Scamper to the nearest vertical edge, fleeing the cursor if present.
            lo = home.left()
            hi = home.right() - self.width()
            center_x = self.x() + self.width() // 2
            mid = (home.left() + home.right()) // 2
            ref = cursor.x() if cursor is not None else center_x
            self._climb_x = hi if ref < mid else lo
            self._facing = 1 if self._climb_x >= self.x() else -1

    def _tick_transit(self, now: float):
        home = self._home_geometry()
        top_y = self._perch_y(home)
        floor_y = self._floor_y(home)
        style = self._profile.transit

        if self._transit == TransitPhase.ASCENDING:
            if style == TransitStyle.TELEPORT_FLOAT:
                # Blink out at the floor, reappear at the top halfway through.
                prev = self._teleport_p
                self._teleport_p = min(1.0, self._teleport_p + TELEPORT_STEP)
                if prev < 0.5 <= self._teleport_p:
                    self.move(self.x(), top_y)
                if self._teleport_p >= 1.0:
                    self._enter_perch(now)
            elif style == TransitStyle.CLIMB:
                # Reach the edge first, then crawl straight up it.
                if abs(self.x() - self._climb_x) > FLY_STEP:
                    step = FLY_STEP if self._climb_x > self.x() else -FLY_STEP
                    self.move(self.x() + step, self.y())
                    return
                self.move(self._climb_x, max(top_y, self.y() - FLY_STEP))
                if self.y() <= top_y:
                    self._enter_perch(now)
            else:  # FLIGHT
                self.move(self.x(), max(top_y, self.y() - FLY_STEP))
                if self.y() <= top_y:
                    self._enter_perch(now)

        elif self._transit == TransitPhase.PERCHED:
            self.move(self.x(), top_y)
            # Come back down once the dwell elapses and the cursor isn't on us.
            if now >= self._perch_until:
                self._transit = TransitPhase.DESCENDING
                self._teleport_p = 0.0
                self._set_state(FamiliarState.WALK)

        elif self._transit == TransitPhase.DESCENDING:
            # Teleporters *float* gently down; flyers/climbers retrace their path.
            bob = 0
            if style == TransitStyle.TELEPORT_FLOAT:
                bob = 1 if (self._frame % 2 == 0) else -1
            ny = min(floor_y, self.y() + FLOAT_STEP)
            self.move(self.x() + bob, ny)
            if ny >= floor_y:
                self._transit = TransitPhase.GROUNDED
                self._fleeing = False
                self._snap_to_floor()
                self._dwell_until = now + random.uniform(DWELL_MIN_S, DWELL_MAX_S)
                self._set_state(FamiliarState.IDLE)

    def _enter_perch(self, now: float):
        self._transit = TransitPhase.PERCHED
        self._teleport_p = 0.0
        self._perch_until = now + random.uniform(PERCH_MIN_S, PERCH_MAX_S)
        self._set_state(FamiliarState.IDLE)

    # ----- rendering --------------------------------------------------------
    def paintEvent(self, event):
        frames = self._active_frames()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        # During a teleport blink, fade the whole sprite out then back in and
        # scatter a sparkle burst tinted with the species' accent colour.
        teleporting = (self._transit == TransitPhase.ASCENDING
                       and self._profile.transit == TransitStyle.TELEPORT_FLOAT)
        if teleporting:
            painter.setOpacity(max(0.0, abs(self._teleport_p - 0.5) * 2.0))

        if frames:
            pm = frames[self._frame % len(frames)]
            x = (self.width() - pm.width()) // 2
            y = (self.height() - pm.height()) // 2
            if self._facing < 0:
                pm = pm.transformed(_flip_transform(), Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap(x, y, pm)
        else:
            # Placeholder art is drawn in a 96px design box, then scaled up to
            # the real sprite size so it stays centred and floor-aligned.
            scale = self.height() / _ART_DESIGN
            painter.save()
            painter.scale(scale, scale)
            self._profile.art.draw(
                painter, self._state, self._frame, self._facing,
                self._transit, _ART_DESIGN, _ART_DESIGN)
            painter.restore()

        if teleporting:
            painter.setOpacity(1.0)
            self._paint_teleport_sparkles(painter)

        painter.end()
        # Inherit the edit-mode dashed border / highlight from the base class.
        super().paintEvent(event)

    def _paint_teleport_sparkles(self, painter: QPainter):
        """Ring of accent-tinted sparkles around the blink point."""
        cx, cy = self.width() // 2, self.height() // 2
        glow = QColor(self._profile.glow)
        # Brightest mid-blink, fading toward the ends.
        glow.setAlpha(int(220 * (1.0 - abs(self._teleport_p - 0.5) * 2.0)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow))
        import math
        ring = 12 + self._teleport_p * 22
        for k in range(8):
            ang = (k / 8.0) * 2 * math.pi + self._teleport_p * 3.0
            sx = cx + int(ring * math.cos(ang))
            sy = cy + int(ring * math.sin(ang))
            r = 2 + (k % 2)
            painter.drawEllipse(QPoint(sx, sy), r, r)

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
                self._transit = TransitPhase.GROUNDED  # a manual drag cancels transit
                self.save_geometry()
                self._snap_to_floor()
            elif self.app is not None and hasattr(self.app, "toggle_chat"):
                self.app.toggle_chat()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def _show_context_menu(self, global_pos: QPoint):
        menu = QMenu()
        change_menu = menu.addMenu(t("familiar.menu.change"))
        for sp in FamiliarSpecies:
            act = change_menu.addAction(t(f"familiar.species.{sp.value}"))
            act.setCheckable(True)
            act.setChecked(sp == self._profile.species)
            act.setData(sp.value)
        menu.addSeparator()
        conjure_action = menu.addAction(t("familiar.menu.conjure"))
        hide_action = menu.addAction(t("familiar.menu.hide"))
        settings_action = menu.addAction(t("familiar.menu.settings"))
        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen == conjure_action and self.app is not None and hasattr(self.app, "conjure_familiar"):
            self.app.conjure_familiar()
        elif chosen == hide_action and self.app is not None and hasattr(self.app, "toggle_familiar"):
            self.app.toggle_familiar()
        elif chosen == settings_action and self.app is not None and hasattr(self.app, "_open_settings"):
            self.app._open_settings()
        elif chosen.data() in {sp.value for sp in FamiliarSpecies}:
            self._apply_species_choice(chosen.data())

    def _apply_species_choice(self, value: str):
        # Picking "Conjure ✨" here switches to the existing custom familiar; use
        # the Conjure dialog (right-click → "Conjure new familiar…", or Settings)
        # to generate a new one.
        self.set_species(value)
        if hasattr(self.app, "settings") and self.app.settings:
            try:
                from mage.settings_keys import KEY_FAMILIAR_TYPE
                self.app.settings.setValue(KEY_FAMILIAR_TYPE, value)
            except Exception:
                pass


def _flip_transform():
    from PyQt6.QtGui import QTransform
    return QTransform().scale(-1, 1)
