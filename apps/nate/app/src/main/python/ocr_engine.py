# MAGE Companion — Android OCR and local dictionary client.
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

import cv2
import numpy as np
from paddleocr import PaddleOCR
import json

class PaddleOCREngine:
    def __init__(self):
        # Initialize PaddleOCR with mobile models for performance on Android
        self.ocr = PaddleOCR(use_angle_cls=True, lang='ch', use_gpu=False,
                             show_log=False)
        print("PaddleOCR Mobile Engine Initialized")

    def detect_and_recognize(self, image_data):
        """
        Processes image data (passed as a bytearray from Kotlin)
        and returns OCR results as a JSON string.
        """
        # Convert bytearray to numpy array for OpenCV
        nparr = np.frombuffer(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return json.dumps([])

        # Run OCR inference
        result = self.ocr.ocr(img, cls=True)
        
        formatted_results = []
        if result and result[0]:
            for line in result[0]:
                # line format: [ [[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence) ]
                box = line[0]
                text, confidence = line[1]
                
                # Convert polygon to [left, top, right, bottom]
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                rect = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                
                formatted_results.append({
                    "text": text,
                    "confidence": float(confidence),
                    "box": rect
                })

        return json.dumps(formatted_results)

_engine = None

def get_engine():
    global _engine
    if _engine is None:
        _engine = PaddleOCREngine()
    return _engine
