# import cv2
# import pytesseract
# import easyocr
# import pytesseract
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"
#
#
# class OCREngine:
#     def __init__(self, engine="easyocr"):
#         """
#         engine: 'easyocr', 'tesseract', or 'auto'
#         """
#         self.engine_name = engine
#         self.reader = None
#
#         if engine in ["auto", "easyocr"]:
#             try:
#                 self.reader = easyocr.Reader(['en'])
#                 self.engine_name = "easyocr"
#                 print("EasyOCR initialized.")
#             except Exception as e:
#                 print("Failed to load EasyOCR:", e)
#                 if engine == "easyocr":
#                     raise
#
#         if engine == "tesseract":
#             self.engine_name = "tesseract"
#
#     def recognize(self, img):
#         results = []
#
#         if self.engine_name == "easyocr":
#             ocr_result = self.reader.readtext(img)
#             for bbox, text, conf in ocr_result:
#                 results.append({
#                     "engine": "easyocr",
#                     "text": text,
#                     "confidence": float(conf),
#                     "bbox": bbox
#                 })
#
#         elif self.engine_name == "tesseract":
#             data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
#             for i in range(len(data["text"])):
#                 if data["text"][i].strip():
#                     results.append({
#                         "engine": "tesseract",
#                         "text": data["text"][i],
#                         "confidence": float(data["conf"][i]),
#                         "bbox": [
#                             data["left"][i],
#                             data["top"][i],
#                             data["width"][i],
#                             data["height"][i]
#                         ]
#                     })
#         return results





import cv2
import pytesseract
import easyocr

import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"


class OCREngine:
    def __init__(self, engine="easyocr", ):
        """
        engine: 'easyocr', 'tesseract', or 'auto'
        """
        self.engine_name = engine
        self.reader = None

        # Try EasyOCR first (if auto or easyocr)
        if engine in ["auto", "easyocr"]:
            try:
                self.reader = easyocr.Reader(['en'])
                self.engine_name = "easyocr"
                print("EasyOCR initialized.")
            except Exception as e:
                print("Failed to load EasyOCR:", e)
                if engine == "easyocr":
                    raise

        # If explicitly Tesseract
        if engine == "tesseract":
            self.engine_name = "tesseract"

    # ------------------------------------------------------------
    # IMPORTANT: NORMALIZE TESSERACT BBOX TO 4-POINT POLYGON
    # ------------------------------------------------------------
    def _convert_tesseract_bbox(self, left, top, width, height):
        """
        Convert Tesseract bbox (left, top, width, height) to:
        [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
        """
        x1, y1 = int(left), int(top)
        x2, y2 = int(left + width), int(top + height)

        return [
            [x1, y1],  # top-left
            [x2, y1],  # top-right
            [x2, y2],  # bottom-right
            [x1, y2],  # bottom-left
        ]

    def recognize(self, img):
        results = []

        # ---------------- EASYOCR ----------------
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

        # ---------------- TESSERACT ----------------
        try:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            n = len(data["text"])
            for i in range(n):
                text = data["text"][i].strip()
                conf = float(data["conf"][i])

                # skip blanks
                if not text:
                    continue
                # skip garbage/confidence -1
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



