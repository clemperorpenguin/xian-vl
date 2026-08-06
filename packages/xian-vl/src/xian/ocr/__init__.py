# Xian-VL — Core Vision-Language orchestration engine.
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

"""Local text detection and recognition, split out from translation.

The vision model can locate and translate text in one call, but it costs
seconds per frame.  Splitting the job — a small local OCR model finds and reads
the lines, the LLM only translates them — turns that into a few hundred
milliseconds, which is the difference between "refreshing" and "live".

This runs in-process on decoded pixels: no PNG encode, no base64, no HTTP.
That is also the only place in MAGE where "direct memory sharing" between a
capture buffer and an accelerator is actually achievable.
"""

from xian.ocr.engine import OcrEngine, OcrLine, ocr_available
from xian.ocr.translate import batch_translate

__all__ = ["OcrEngine", "OcrLine", "ocr_available", "batch_translate"]
