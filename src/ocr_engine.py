import cv2
import pytesseract
import easyocr

import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


class OCREngine:
    def __init__(self, engine="easyocr", ):
        
        self.engine_name = engine
        self.reader = None

        if engine in ["auto", "easyocr"]:
            try:
                self.reader = easyocr.Reader(['en'])
                self.engine_name = "easyocr"
                print("EasyOCR initialized.")
            except Exception as e:
                print("Failed to load EasyOCR:", e)
                if engine == "easyocr":
                    raise

        if engine == "tesseract":
            self.engine_name = "tesseract"

    def _convert_tesseract_bbox(self, left, top, width, height):
        x1, y1 = int(left), int(top)
        x2, y2 = int(left + width), int(top + height)

        return [
            [x1, y1],  
            [x2, y1],  
            [x2, y2],  
            [x1, y2],  
        ]

    def recognize(self, img):
        results = []

        try:
            easy = self.reader.readtext(img)
            for bbox, text, conf in easy:
                results.append({
                    "engine": "easyocr",
                    "text": text,
                    "confidence": float(conf),
                    "bbox": [[int(x), int(y)] for x, y in bbox]
                })
        except Exception as e:
            print("EasyOCR failed:", e)

        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])

                if not text:
                    continue
        
                if conf < 0:
                    continue

                bbox = self._convert_tesseract_bbox(
                    data["left"][i],
                    data["top"][i],
                    data["width"][i],
                    data["height"][i]
                )

                results.append({
                    "engine": "tesseract",
                    "text": text,
                    "confidence": conf,
                    "bbox": bbox
                })
        except Exception as e:
            print("Tesseract failed:", e)

        return results



