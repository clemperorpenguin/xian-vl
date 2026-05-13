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
