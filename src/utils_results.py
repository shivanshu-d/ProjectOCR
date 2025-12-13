import json
from pathlib import Path
from datetime import datetime
import cv2


def save_result_bundle(
    image_name: str,
    extracted_id: str,
    extractor_result: dict,
    visual_image_rgb,
    confidence: float = None,
):
    """
    Save extraction results and OCR visualization.

    Output structure:
    results/
      <image_name_without_ext>/
        ├── result.json
        └── ocr_visual.png
    """

    base_name = Path(image_name).stem
    out_dir = Path("results") / base_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Save JSON ----------------
    result_json = {
        "image": image_name,
        "extracted_id": extracted_id,
        "confidence": confidence,
        "success": extractor_result.get("success"),
        "matched_line": extractor_result.get("matched_line"),
        "all_lines": extractor_result.get("all_lines"),
        "timestamp": datetime.now().isoformat(),
    }

    json_path = out_dir / "result.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_json, f, indent=2)

    # ---------------- Save Image ----------------
    img_path = out_dir / "ocr_visual.png"
    # visual_image_rgb is RGB → convert to BGR for OpenCV
    cv2.imwrite(str(img_path), cv2.cvtColor(visual_image_rgb, cv2.COLOR_RGB2BGR))

    return json_path, img_path
