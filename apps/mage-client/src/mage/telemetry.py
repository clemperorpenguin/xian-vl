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

"""In-process performance telemetry for MAGE.

Memory-first by design: bounded ring buffers accumulate two streams while the
app is used —

* **Inference timings** (TTFT, throughput) lifted from the ``/v1/stats`` payload
  MAGE already fetches after every request — previously logged and discarded.
* **Host resource utilisation** (CPU / GPU / RAM / VRAM) sampled on an interval.

VRAM and GPU load are sourced from **Lemonade's own accounting** first
(``/v1/system-info`` + ``/v1/stats``) — the same numbers a dynamic VRAM-eviction
policy reasons about — with a best-effort ``amd-smi`` / ``rocm-smi`` fallback.
CPU and RAM come from ``psutil`` when present. Every source degrades to ``None``
rather than raising, so telemetry never affects the overlay's hot path.

Nothing is written to disk unless :meth:`TelemetryRecorder.enable_jsonl` is
called. The accumulated data powers the in-app "Performance Report" export and
doubles as the real-world validation harness for the Lemonade VRAM work.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Optional

logger = logging.getLogger(__name__)

_MB = 1024 * 1024


# --------------------------------------------------------------------------- #
# Defensive numeric extraction
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> Optional[float]:
    """Coerce to float, tolerating strings like ``"4096 MiB"`` / ``"37%"``."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = []
        seen_dot = False
        for ch in value.strip():
            if ch.isdigit():
                cleaned.append(ch)
            elif ch == "." and not seen_dot:
                cleaned.append(ch)
                seen_dot = True
            elif ch in "-+" and not cleaned:
                cleaned.append(ch)
            elif cleaned:
                break
        try:
            return float("".join(cleaned)) if cleaned else None
        except ValueError:
            return None
    return None


def _find_first(data: Any, tokens: Iterable[str]) -> Optional[float]:
    """Recursively search a JSON-ish structure for the first numeric value whose
    key contains any of ``tokens`` (case-insensitive). Lemonade's exact field
    names vary by version/backend, so we match loosely rather than hard-code."""
    toks = tuple(t.lower() for t in tokens)
    if isinstance(data, dict):
        for key, val in data.items():
            if isinstance(key, str) and any(t in key.lower() for t in toks):
                num = _to_float(val)
                if num is not None:
                    return num
        for val in data.values():
            found = _find_first(val, toks)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_first(item, toks)
            if found is not None:
                return found
    return None


def _normalize_vram(sample: ResourceSample) -> None:
    """Heuristic: some backends report VRAM in bytes. Anything implausibly large
    for a "MB" reading (> 512 GB) is almost certainly bytes, so rescale it."""
    for attr in ("vram_used_mb", "vram_total_mb"):
        v = getattr(sample, attr)
        if v is not None and v > 512_000:
            setattr(sample, attr, v / _MB)


# --------------------------------------------------------------------------- #
# Sample records
# --------------------------------------------------------------------------- #
@dataclass
class InferenceSample:
    ts: float
    modality: str
    model: Optional[str]
    ttft_ms: Optional[float]
    output_tokens: Optional[float]
    tokens_per_sec: Optional[float]


@dataclass
class ResourceSample:
    ts: float
    cpu_pct: Optional[float] = None
    ram_used_mb: Optional[float] = None
    ram_total_mb: Optional[float] = None
    gpu_pct: Optional[float] = None
    vram_used_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None

    def is_empty(self) -> bool:
        return all(
            getattr(self, f) is None
            for f in ("cpu_pct", "ram_used_mb", "gpu_pct", "vram_used_mb")
        )


def parse_inference_stats(stats: dict, modality: str, model: Optional[str]) -> InferenceSample:
    """Map a ``/v1/stats`` payload onto an :class:`InferenceSample`."""
    ttft = _find_first(stats, ("time_to_first_token", "ttft", "first_token", "prompt_eval"))
    # Lemonade reports TTFT in seconds; normalise to ms when it looks sub-second.
    if ttft is not None and ttft < 100:
        ttft *= 1000.0
    return InferenceSample(
        ts=time.time(),
        modality=modality,
        model=model,
        ttft_ms=ttft,
        output_tokens=_find_first(stats, ("output_tokens", "completion_tokens", "generated_tokens", "decode_token_count")),
        tokens_per_sec=_find_first(stats, ("tokens_per_second", "output_tokens_per_second", "decode_token_per_second", "throughput", "tps")),
    )


def parse_lemonade_resources(system_info: Any, stats: Any) -> ResourceSample:
    """Pull GPU load + VRAM out of Lemonade's hardware/stat payloads."""
    sample = ResourceSample(ts=time.time())
    for src in (stats, system_info):
        if sample.vram_used_mb is None:
            sample.vram_used_mb = _find_first(src, ("vram_used", "vram_in_use", "gpu_memory_used", "memory_used"))
        if sample.vram_total_mb is None:
            # Note: no bare "vram" token here — it would also match "vram_used".
            sample.vram_total_mb = _find_first(src, ("vram_total", "total_vram", "gpu_memory_total", "memory_total"))
        if sample.gpu_pct is None:
            sample.gpu_pct = _find_first(src, ("gpu_util", "gpu_load", "gpu_usage", "gpu_busy", "utilization"))
    _normalize_vram(sample)
    return sample


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #
class TelemetryRecorder:
    """Thread-safe, memory-first store for inference + resource samples."""

    def __init__(self, capacity: int = 5000):
        self._lock = threading.Lock()
        self._inf: deque[InferenceSample] = deque(maxlen=capacity)
        self._res: deque[ResourceSample] = deque(maxlen=capacity)
        self._jsonl_path: Optional[str] = None
        self.session_start = time.time()

    # -- ingest ----------------------------------------------------------- #
    def record_inference(self, stats: dict, modality: str, model: Optional[str] = None) -> None:
        try:
            sample = parse_inference_stats(stats, modality, model)
        except Exception:  # never let telemetry break an inference path
            logger.debug("Telemetry: failed to parse inference stats", exc_info=True)
            return
        with self._lock:
            self._inf.append(sample)
        self._maybe_persist("inference", sample)

    def record_resources(self, sample: ResourceSample) -> None:
        if sample is None or sample.is_empty():
            return
        with self._lock:
            self._res.append(sample)
        self._maybe_persist("resource", sample)

    # -- disk (opt-in) ---------------------------------------------------- #
    def enable_jsonl(self, path: str) -> None:
        """Begin mirroring every sample to a JSONL file (disk is optional)."""
        self._jsonl_path = path
        logger.info("Telemetry: JSONL persistence enabled at %s", path)

    def _maybe_persist(self, kind: str, sample: Any) -> None:
        if not self._jsonl_path:
            return
        try:
            with open(self._jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps({"kind": kind, **asdict(sample)}) + "\n")
        except Exception:
            logger.debug("Telemetry: JSONL write failed", exc_info=True)

    # -- read ------------------------------------------------------------- #
    def clear(self) -> None:
        with self._lock:
            self._inf.clear()
            self._res.clear()
            self.session_start = time.time()

    def snapshot(self) -> tuple[list[InferenceSample], list[ResourceSample]]:
        with self._lock:
            return list(self._inf), list(self._res)

    # -- report ----------------------------------------------------------- #
    def to_markdown(self) -> str:
        inf, res = self.snapshot()
        elapsed = time.time() - self.session_start
        lines = [
            "# MAGE Performance Report",
            "",
            f"- Session: **{_fmt_duration(elapsed)}**",
            f"- Inference samples: **{len(inf)}**  ·  Resource samples: **{len(res)}**",
            "",
        ]

        # Per-modality inference table
        lines.append("## Inference (per modality)")
        lines.append("")
        if inf:
            lines.append("| Modality | N | TTFT p50 | TTFT p95 | Tok/s mean | Out-tok mean |")
            lines.append("|---|---|---|---|---|---|")
            modalities: dict[str, list[InferenceSample]] = {}
            for s in inf:
                modalities.setdefault(s.modality, []).append(s)
            for mod in sorted(modalities):
                rows = modalities[mod]
                ttfts = [r.ttft_ms for r in rows]
                lines.append(
                    f"| {mod} | {len(rows)} "
                    f"| {_fmt_ms(_pct(ttfts, 50))} | {_fmt_ms(_pct(ttfts, 95))} "
                    f"| {_fmt(_mean([r.tokens_per_sec for r in rows]))} "
                    f"| {_fmt(_mean([r.output_tokens for r in rows]))} |"
                )
        else:
            lines.append("_No inference recorded yet._")
        lines.append("")

        # Resource utilisation
        lines.append("## Resource utilisation (CPU / GPU / RAM / VRAM)")
        lines.append("")
        if res:
            lines.append("| Metric | Mean | Max |")
            lines.append("|---|---|---|")
            lines.append(f"| CPU % | {_fmt(_mean([r.cpu_pct for r in res]))} | {_fmt(_max([r.cpu_pct for r in res]))} |")
            lines.append(f"| GPU % | {_fmt(_mean([r.gpu_pct for r in res]))} | {_fmt(_max([r.gpu_pct for r in res]))} |")
            lines.append(f"| RAM used (GB) | {_fmt_gb(_mean([r.ram_used_mb for r in res]))} | {_fmt_gb(_max([r.ram_used_mb for r in res]))} |")
            lines.append(f"| VRAM used (GB) | {_fmt_gb(_mean([r.vram_used_mb for r in res]))} | {_fmt_gb(_max([r.vram_used_mb for r in res]))} |")
            vram_total = _max([r.vram_total_mb for r in res])
            ram_total = _max([r.ram_total_mb for r in res])
            if vram_total is not None:
                lines.append(f"| VRAM total (GB) | — | {_fmt_gb(vram_total)} |")
            if ram_total is not None:
                lines.append(f"| RAM total (GB) | — | {_fmt_gb(ram_total)} |")
        else:
            lines.append("_No resource samples yet (server unreachable or sampler disabled)._")
        lines.append("")
        lines.append("> CPU/RAM via psutil; GPU/VRAM via Lemonade `/v1/system-info` + `/v1/stats` "
                     "(amd-smi/rocm-smi fallback).")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Aggregation helpers (None-tolerant)
# --------------------------------------------------------------------------- #
def _clean(values: Iterable[Optional[float]]) -> list[float]:
    return [v for v in values if v is not None]


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = _clean(values)
    return sum(vals) / len(vals) if vals else None


def _max(values: Iterable[Optional[float]]) -> Optional[float]:
    vals = _clean(values)
    return max(vals) if vals else None


def _pct(values: Iterable[Optional[float]], p: float) -> Optional[float]:
    vals = sorted(_clean(values))
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    rank = (p / 100.0) * (len(vals) - 1)
    low = int(rank)
    frac = rank - low
    if low + 1 < len(vals):
        return vals[low] + frac * (vals[low + 1] - vals[low])
    return vals[low]


def _fmt(v: Optional[float]) -> str:
    return f"{v:.1f}" if v is not None else "—"


def _fmt_ms(v: Optional[float]) -> str:
    return f"{v:.0f} ms" if v is not None else "—"


def _fmt_gb(mb: Optional[float]) -> str:
    return f"{mb / 1024:.2f}" if mb is not None else "—"


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# --------------------------------------------------------------------------- #
# Process-wide singleton
# --------------------------------------------------------------------------- #
_recorder: Optional[TelemetryRecorder] = None


def get_recorder() -> TelemetryRecorder:
    global _recorder
    if _recorder is None:
        _recorder = TelemetryRecorder()
    return _recorder


# --------------------------------------------------------------------------- #
# Sampler — driven by a main-thread QTimer, work runs on the async engine
# --------------------------------------------------------------------------- #
class TelemetrySampler:
    """Collects one :class:`ResourceSample` per :meth:`tick`.

    :meth:`tick` is cheap by design — it only dispatches a coroutine to the
    shared :class:`AsyncEngine`, so psutil reads, the Lemonade poll, and the
    optional GPU-SMI subprocess all stay off the PyQt6 GUI thread.
    """

    def __init__(self, processor, recorder: Optional[TelemetryRecorder] = None):
        self.processor = processor
        self.recorder = recorder or get_recorder()
        self._gpu_probe_disabled = False
        try:
            import psutil  # noqa: F401
            self._psutil = psutil
            self._psutil.cpu_percent(None)  # prime the delta baseline
        except Exception:
            self._psutil = None
            logger.info("Telemetry: psutil unavailable; CPU/RAM will be omitted.")

    def tick(self) -> None:
        engine = getattr(self.processor, "engine", None)
        if engine is None:
            return
        try:
            engine.submit(self._sample())
        except Exception:
            logger.debug("Telemetry: sample dispatch failed", exc_info=True)

    async def _sample(self) -> None:
        sample = ResourceSample(ts=time.time())

        # Local CPU / RAM (psutil) — fast, non-blocking.
        if self._psutil is not None:
            try:
                sample.cpu_pct = self._psutil.cpu_percent(None)
                vm = self._psutil.virtual_memory()
                sample.ram_used_mb = vm.used / _MB
                sample.ram_total_mb = vm.total / _MB
            except Exception:
                logger.debug("Telemetry: psutil sample failed", exc_info=True)

        # GPU / VRAM from Lemonade's own accounting (the eviction-policy source).
        try:
            from xian.lemonade_client import LemonadeClient
            base = getattr(self.processor.config, "api_url", "http://localhost:13305/v1")
            async with LemonadeClient(base_url=base.removesuffix("/v1")) as client:
                sysinfo: Any = {}
                stats: Any = {}
                try:
                    sysinfo = await client.system_info()
                except Exception:
                    pass
                try:
                    stats = await client.stats()
                except Exception:
                    pass
            lem = parse_lemonade_resources(sysinfo, stats)
            sample.gpu_pct = lem.gpu_pct
            sample.vram_used_mb = lem.vram_used_mb
            sample.vram_total_mb = lem.vram_total_mb
        except Exception:
            logger.debug("Telemetry: Lemonade resource poll failed", exc_info=True)

        # Best-effort AMD SMI fallback when Lemonade exposed no VRAM figure.
        if sample.vram_used_mb is None and not self._gpu_probe_disabled:
            await self._probe_amd_smi(sample)

        self.recorder.record_resources(sample)

    async def _probe_amd_smi(self, sample: ResourceSample) -> None:
        """Try ``amd-smi`` then ``rocm-smi`` for VRAM/GPU. Disables itself after
        a failure so we don't spawn a doomed subprocess every tick."""
        import asyncio
        import shutil

        for binary, args in (
            ("amd-smi", ["metric", "-g", "0", "--mem-usage", "--usage", "--json"]),
            ("rocm-smi", ["--showmeminfo", "vram", "--showuse", "--json"]),
        ):
            if shutil.which(binary) is None:
                continue
            try:
                proc = await asyncio.create_subprocess_exec(
                    binary, *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                out, _ = await asyncio.wait_for(proc.communicate(), timeout=2.0)
                data = json.loads(out.decode("utf-8", "replace"))
            except Exception:
                logger.debug("Telemetry: %s probe failed", binary, exc_info=True)
                continue
            sample.vram_used_mb = _find_first(data, ("vram_used", "used_vram", "vram_total_used", "mem_used"))
            sample.vram_total_mb = _find_first(data, ("vram_total", "total_vram", "mem_total"))
            sample.gpu_pct = sample.gpu_pct or _find_first(data, ("gfx_activity", "gpu_use", "gpu_util", "usage"))
            _normalize_vram(sample)
            if sample.vram_used_mb is not None:
                return
        # Nothing usable; stop trying for the rest of the session.
        self._gpu_probe_disabled = True
