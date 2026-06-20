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

"""Manual system-audio probe (not a pytest test).

Run directly to confirm system-audio capture works on your machine:

    uv run --package mage-client python apps/mage-client/manual_audio_probe.py

Mirrors ContinuousAudioStreamer: uses `parec` (raw PCM to stdout). Play some
audio through your speakers while it runs — RMS should rise above the noise floor.
"""

import asyncio
import struct
import math
import subprocess


async def run_audio_test():
    proc = await asyncio.create_subprocess_exec(
        "parec", "--rate", "16000", "--channels", "1", "--format", "s16le", "--device", "@DEFAULT_MONITOR@",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    print("Reading 50 chunks of 100ms...")
    for i in range(50):
        try:
            data = await asyncio.wait_for(proc.stdout.readexactly(3200), timeout=1.0)
            samples = struct.unpack(f'<{len(data)//2}h', data)
            rms = math.sqrt(sum(s*s for s in samples) / len(samples))
            print(f"Chunk {i}: RMS = {rms:.2f}, max = {max(samples)}, min = {min(samples)}")
        except Exception as e:
            print("Error reading:", e)
            break

    proc.terminate()


if __name__ == "__main__":
    asyncio.run(run_audio_test())
