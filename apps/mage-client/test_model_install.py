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


# ── the translation model chosen in Settings ─────────────────────────

def _settings_app(translation_model, installed):
    from mage.settings_keys import KEY_TRANSLATION_MODEL

    app = SimpleNamespace(
        settings=_Settings({KEY_TRANSLATION_MODEL: translation_model}),
        processor=SimpleNamespace(router=SimpleNamespace(downloaded_model_ids=list(installed))),
        pulls=[],
    )
    app._pull_model = lambda name, body=None: app.pulls.append((name, body))
    return app


def test_choosing_the_npu_model_registers_and_downloads_it():
    """It is not in Lemonade's catalogue: a pull by name alone finds nothing."""
    from xian.translate import NPU_TRANSLATION_MODEL

    app = _settings_app(NPU_TRANSLATION_MODEL, ["Hy-MT2-1.8B-GGUF-Q4_K_M"])
    XianApp._install_translation_model_if_missing(app)

    [(name, body)] = app.pulls
    assert name == NPU_TRANSLATION_MODEL
    assert body["recipe"] == "flm"


def test_an_installed_translation_model_is_left_alone():
    app = _settings_app("Hy-MT2-1.8B-GGUF-Q4_K_M", ["Hy-MT2-1.8B-GGUF-Q4_K_M"])
    XianApp._install_translation_model_if_missing(app)
    assert app.pulls == []


def test_an_unasked_server_is_not_taken_to_be_empty():
    from xian.translate import NPU_TRANSLATION_MODEL

    app = _settings_app(NPU_TRANSLATION_MODEL, [])
    XianApp._install_translation_model_if_missing(app)
    assert app.pulls == []
