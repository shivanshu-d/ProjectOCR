import re
from typing import List, Dict, Any


class TextExtractor:

    def __init__(self, y_threshold: int = 25):
        self.y_threshold = y_threshold
        self.full_id_regex = re.compile(r"\b\d{12,}_1(_[a-zA-Z]+)?\b")
        self.long_number_regex = re.compile(r"\b\d{12,}\b")

    def extract_target(self, ocr_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        tokens = self._normalize_tokens(ocr_results
        direct = self._find_direct_id(tokens)
        if direct:
            return self._success(direct, direct, tokens)
        numeric = self._find_numeric_candidate(tokens)
        if numeric:
            reconstructed = f"{numeric}_1_"
            return self._success(reconstructed, reconstructed, tokens)

        return self._fail(tokens)

    # Token Normalization
   
    def _normalize_tokens(self, ocr_results):
        tokens = []
        for r in ocr_results:
            text = (r.get("text") or "").strip()
            bbox = r.get("bbox")

            if not text or not bbox:
                continue

            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            w = max(xs) - min(xs)
            h = max(ys) - min(ys)

            tokens.append({
                "text": text,
                "clean": text.replace(" ", "").replace("-", "_"),
                "bbox": bbox,
                "w": w,
                "h": h,
                "aspect": w / (h + 1e-6),
                "confidence": float(r.get("confidence", 0))
            })
        return tokens

    def _find_direct_id(self, tokens):
        for t in tokens:
            m = self.full_id_regex.search(t["clean"])
            if m:
                return m.group(0)
        return None

    def _find_numeric_candidate(self, tokens):
        candidates = []

        for t in tokens:
            txt = t["clean"]

            if not self.long_number_regex.fullmatch(txt):
                continue

            if t["h"] > 200 and t["aspect"] < 0.4:
                continue
            if t["w"] > 300 and t["aspect"] > 6:
                continue

            candidates.append(txt)

        if not candidates:
            return None

        candidates.sort(key=len, reverse=True)
        return candidates[0]

    def _success(self, extracted, matched_line, tokens):
        return {
            "success": True,
            "extracted_text": extracted,
            "matched_line": matched_line,
            "all_lines": [t["text"] for t in tokens],
            "raw_groups": None
        }

    def _fail(self, tokens):
        return {
            "success": False,
            "extracted_text": None,
            "matched_line": None,
            "all_lines": [t["text"] for t in tokens],
            "raw_groups": None
        }
