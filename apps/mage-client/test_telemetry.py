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

"""Unit tests for the in-app performance telemetry recorder.

These cover the pure data layer (parsing, aggregation, report rendering) and
need neither a running Lemonade server nor psutil — so they run anywhere.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from mage.telemetry import (  # noqa: E402
    ResourceSample,
    TelemetryRecorder,
    _find_first,
    _pct,
    _to_float,
    parse_inference_stats,
    parse_lemonade_resources,
)


def test_to_float_tolerates_units():
    assert _to_float(42) == 42.0
    assert _to_float("4096 MiB") == 4096.0
    assert _to_float("37%") == 37.0
    assert _to_float("-1.5 ms") == -1.5
    assert _to_float(None) is None
    assert _to_float(True) is None  # bools are not numbers here
    assert _to_float("n/a") is None


def test_find_first_recurses_and_matches_loosely():
    blob = {"gpu": {"details": [{"vram_used_mb": "2048 MiB"}]}}
    assert _find_first(blob, ("vram_used",)) == 2048.0
    assert _find_first(blob, ("nonexistent",)) is None


def test_parse_inference_normalises_ttft_to_ms():
    # Sub-second TTFT reported in seconds → promoted to ms.
    s = parse_inference_stats(
        {"time_to_first_token": 0.25, "tokens_per_second": 80, "output_tokens": 120},
        "vision", "Qwen3.5-4B",
    )
    assert s.ttft_ms == 250.0
    assert s.tokens_per_sec == 80.0
    assert s.output_tokens == 120.0
    assert s.modality == "vision"
    assert s.model == "Qwen3.5-4B"


def test_parse_lemonade_resources_bytes_to_mb():
    # VRAM reported in bytes (8 GiB) is rescaled to MB.
    sysinfo = {"gpu": {"vram_total": 8 * 1024 * 1024 * 1024}}
    stats = {"vram_used": 4 * 1024 * 1024 * 1024, "gpu_util": 55}
    r = parse_lemonade_resources(sysinfo, stats)
    assert abs(r.vram_total_mb - 8192) < 1
    assert abs(r.vram_used_mb - 4096) < 1
    assert r.gpu_pct == 55.0


def test_percentile():
    vals = [10, 20, 30, 40, 50]
    assert _pct(vals, 50) == 30
    assert _pct([], 50) is None
    assert _pct([7], 95) == 7


def test_recorder_accumulates_and_reports():
    rec = TelemetryRecorder()
    rec.record_inference({"ttft": 0.1, "tokens_per_second": 90, "output_tokens": 50}, "vision", "m")
    rec.record_inference({"ttft": 0.2, "tokens_per_second": 70, "output_tokens": 60}, "asr", "m")
    rec.record_resources(ResourceSample(ts=0, cpu_pct=20, ram_used_mb=8000, vram_used_mb=4096))

    inf, res = rec.snapshot()
    assert len(inf) == 2 and len(res) == 1

    md = rec.to_markdown()
    assert "# MAGE Performance Report" in md
    assert "vision" in md and "asr" in md
    assert "Resource utilisation" in md


def test_empty_resource_sample_is_dropped():
    rec = TelemetryRecorder()
    rec.record_resources(ResourceSample(ts=0))  # all None
    _, res = rec.snapshot()
    assert res == []


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"\nALL {len(fns)} TELEMETRY TESTS PASSED")
