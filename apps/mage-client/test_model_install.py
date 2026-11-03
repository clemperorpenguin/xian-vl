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

"""When the health check downloads the configured model, and when it does not.

A collection is tens of gigabytes and the local OCR path never touches it, so
finding one missing must not start a download unless the user just asked for
one — at first run, or by picking a tier in Settings.
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mage.app import XianApp  # noqa: E402
from mage.settings_keys import KEY_API_MODEL  # noqa: E402

INSTALLED = [{"id": "Hy-MT2-1.8B-GGUF-Q4_K_M", "downloaded": True}]


class _Settings(dict):
    def value(self, key, default=None):
        return self.get(key, default)

    def setValue(self, key, value):
        self[key] = value


def _app(model="Xian-Halo", *, requested=False):
    router = SimpleNamespace(update_with_models=lambda models: None, omni_detected=False, active_model=None)
    app = SimpleNamespace(
        settings=_Settings({KEY_API_MODEL: model}),
        processor=SimpleNamespace(router=router, config=SimpleNamespace(model_name=model)),
        _model_pull_requested=requested,
        _missing_model=None,
        pulls=[],
    )
    app._pull_model = lambda name, body=None: app.pulls.append(name)
    return app


def _health(app, raw=INSTALLED):
    XianApp._on_health_result(app, True, [m["id"] for m in raw], raw)


def test_a_missing_model_is_reported_not_downloaded():
    app = _app()
    _health(app)
    assert app.pulls == []
    assert app._missing_model == "Xian-Halo"


def test_repeated_health_checks_still_do_not_download():
    app = _app()
    _health(app)
    _health(app)
    assert app.pulls == []


def test_a_requested_model_is_downloaded_once():
    app = _app(requested=True)
    _health(app)
    _health(app)
    assert app.pulls == ["Xian-Halo"]
    assert app._model_pull_requested is False


def test_an_installed_model_clears_the_missing_note():
    app = _app()
    _health(app)
    _health(app, INSTALLED + [{"id": "Xian-Halo", "downloaded": True}])
    assert app._missing_model is None
    assert app.pulls == []
