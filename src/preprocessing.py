# import cv2
# import numpy as np
#
# class Preprocessor:
#
#     def __init__(
#         self,
#         do_denoise=True,
#         do_clahe=True,
#         do_threshold=True,
#         do_sharpen=True,
#         do_deskew=False,
#         target_width=1200
#     ):
#         self.do_denoise = do_denoise
#         self.do_clahe = do_clahe
#         self.do_threshold = do_threshold
#         self.do_sharpen = do_sharpen
#         self.do_deskew = do_deskew
#         self.target_width = target_width
#
#     def resize(self, image):
#         h, w = image.shape[:2]
#         scale = self.target_width / w
#         new_h = int(h * scale)
#         return cv2.resize(image, (self.target_width, new_h), interpolation=cv2.INTER_CUBIC)
#
#     def to_gray(self, image):
#         if len(image.shape) == 3:
#             return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         return image
#
#     def denoise(self, image):
#         return cv2.fastNlMeansDenoising(image, h=15)
#
#     def clahe(self, image):
#         clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#         return clahe.apply(image)
#
#     def threshold(self, image):
#         return cv2.adaptiveThreshold(
#             image, 255,
#             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#             cv2.THRESH_BINARY,
#             25, 15
#         )
#
#     def sharpen(self, image):
#         gaussian = cv2.GaussianBlur(image, (5, 5), 2.0)
#         return cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
#
#     def deskew(self, image):
#         gray = self.to_gray(image)
#         thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
#         coords = np.column_stack(np.where(thresh > 0))
#         angle = cv2.minAreaRect(coords)[-1]
#         if angle < -45:
#             angle = -(90 + angle)
#         else:
#             angle = -angle
#         h, w = gray.shape[:2]
#         M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
#         return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
#
#     def run_pipeline(self, image):
#         img = self.resize(image)
#         img = self.to_gray(img)
#         if self.do_denoise:
#             img = self.denoise(img)
#         if self.do_clahe:
#             img = self.clahe(img)
#         if self.do_threshold:
#             img = self.threshold(img)
#         if self.do_sharpen:
#             img = self.sharpen(img)
#         if self.do_deskew:
#             img = self.deskew(img)
#         return img

import cv2
import numpy as np

import cv2
import numpy as np

class Preprocessor:

    def __init__(
        self,
        do_denoise=True,
        do_clahe=True,
        do_threshold=False,  # OFF by default (can destroy barcode)
        do_sharpen=False,    # OFF by default
        do_deskew=False,
        target_width=1200
    ):
        self.do_denoise = do_denoise
        self.do_clahe = do_clahe
        self.do_threshold = do_threshold
        self.do_sharpen = do_sharpen
        self.do_deskew = do_deskew
        self.target_width = target_width

    def resize(self, image):
        h, w = image.shape[:2]
        if w == 0:
            return image
        scale = self.target_width / float(w)
        new_h = max(1, int(h * scale))
        return cv2.resize(image, (self.target_width, new_h), interpolation=cv2.INTER_CUBIC)

    def to_gray(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    def denoise(self, img):
        # mild denoising (preserves text edges)
        if len(img.shape) == 2:
            return cv2.fastNlMeansDenoising(img, h=10)
        else:
            return cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)

    def clahe(self, img):
        gray = self.to_gray(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        eq = clahe.apply(gray)
        return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)

    def threshold(self, img):
        gray = self.to_gray(img)
        return cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 9
        )

    def sharpen(self, img):
        gray = self.to_gray(img)
        blur = cv2.GaussianBlur(gray, (5, 5), 1)
        sharp = cv2.addWeighted(gray, 1.2, blur, -0.2, 0)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    def deskew(self, img):
        gray = self.to_gray(img)
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        coords = np.column_stack(np.where(th > 0))
        if coords.size == 0:
            return img
        angle = cv2.minAreaRect(coords)[-1]
        angle = -(90 + angle) if angle < -45 else -angle
        h, w = gray.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

    def run_pipeline(self, img):
        img = self.resize(img)

        if self.do_denoise:
            img = self.denoise(img)

        if self.do_clahe:
            img = self.clahe(img)

        if self.do_threshold:
            img = self.threshold(img)

        if self.do_sharpen:
            img = self.sharpen(img)

        if self.do_deskew:
            img = self.deskew(img)

        return img
