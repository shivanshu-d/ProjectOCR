# app.py
"""
Streamlit app for OCR-based `_1_` ID extraction.
Usage:
    streamlit run app.py

Requirements (from your repo):
    - src/preprocessing.py  (Preprocessor)
    - src/ocr_engine.py     (OCREngine)
    - src/text_extraction.py (TextExtractor)

The app lets you:
 - Upload an image (or pick a sample)
 - Configure basic preprocessing toggles
 - Choose OCR engine (easyocr / tesseract / auto)
 - Run OCR and extract the `_1_` ID token
 - Visualize OCR bounding boxes and highlight the extracted token
 - Download JSON result for the image
"""

import json
import io
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
import numpy as np
import cv2
from src.utils_results import save_result_bundle

# Ensure project src is importable
import sys
sys.path.append(".")

# Import your modules
from src.preprocessing import Preprocessor
from src.ocr_engine import OCREngine
from src.text_extraction import TextExtractor

# ---------- Helpers ----------
def draw_bboxes_on_image(img: np.ndarray, ocr_results: List[Dict[str, Any]], highlight_token: str = None) -> np.ndarray:
    """
    Draw bounding boxes on a BGR image. Highlight the box(es) containing highlight_token.
    Returns an RGB image ready for st.image (Streamlit expects RGB).
    """
    out = img.copy()
    # If grayscale convert to BGR
    if len(out.shape) == 2:
        out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

    for r in ocr_results:
        try:
            bbox = r["bbox"]
            text = r.get("text", "")
            conf = r.get("confidence", None)

            x_coords = [int(pt[0]) for pt in bbox]
            y_coords = [int(pt[1]) for pt in bbox]
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)

            # Choose color: highlight if token contains highlight_token, else green
            color = (0, 255, 0)  # green in BGR
            if highlight_token:
                # compare normalized token
                norm_text = text.replace(" ", "").replace("-", "_")
                if highlight_token.replace(" ", "").replace("-", "_") in norm_text:
                    color = (0, 120, 255)  # orange-ish (BGR)
                    thickness = 3
                else:
                    thickness = 1
            else:
                thickness = 1

            cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
            label = text if len(text) < 30 else text[:27] + "..."
            cv2.putText(out, label, (x1, max(10, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        except Exception:
            # ignore malformed bbox
            continue

    # Convert BGR->RGB for display
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def ocr_results_confidence_for_token(ocr_results, extracted_token):
    """
    Better confidence calculation:
    Match prefix, '1', and suffix separately.
    Average confidence of matching OCR tokens.
    """
    if not extracted_token:
        return 0.0

    # Split token into prefix, middle, suffix
    parts = extracted_token.split("_")
    parts = [p.strip().lower() for p in parts if p.strip()]

    confidences = []

    for r in ocr_results:
        t = r.get("text", "").replace(" ", "").replace("-", "").lower()
        conf = r.get("confidence", None)

        if conf is None:
            continue

        try:
            conf = float(conf)
        except:
            continue

        # check if token contains any part (prefix or suffix)
        for p in parts:
            if p and p in t:
                confidences.append(conf)
                break

    if not confidences:
        return 0.0

    return sum(confidences) / len(confidences)



# ---------- Streamlit UI ----------
st.set_page_config(page_title="Waybill `_1_` Extractor", layout="wide")

st.title("Waybill / Shipping Label — `_1_` ID Extractor")
st.write("Upload a shipping label image. The app will preprocess, OCR, and extract the ID token containing `_1_` (e.g. `160390797970200578_1_gsm`).")

# Sidebar: options
st.sidebar.header("Settings")

# Preprocessing toggles
st.sidebar.subheader("Preprocessing")
do_denoise = st.sidebar.checkbox("Denoise", value=True)
do_clahe = st.sidebar.checkbox("CLAHE (contrast)", value=True)
do_threshold = st.sidebar.checkbox("Adaptive Threshold", value=False, help="Tends to break barcodes / dense numeric lines; keep OFF unless necessary.")
do_sharpen = st.sidebar.checkbox("Sharpen", value=True)
do_deskew = st.sidebar.checkbox("Deskew", value=False)
target_width = st.sidebar.number_input("Target width (px)", min_value=600, max_value=3000, value=1200, step=100)

# OCR settings
st.sidebar.subheader("OCR Engine")
engine = st.sidebar.radio("Engine", options=["easyocr", "tesseract", "auto"], index=0)
st.sidebar.markdown("**Note:** `easyocr` typically performs better on noisy label text. `tesseract` may need a valid system installation.")

# Upload image
st.sidebar.subheader("Sample / Upload")
sample_images_dir = Path("datasets/ReverseWay Bill")
sample_paths = list(sample_images_dir.glob("*.*")) if sample_images_dir.exists() else []
sample_files = [p.name for p in sample_paths][:10]
selected_sample = st.sidebar.selectbox("Pick sample image (or upload below)", options=["-- none --"] + sample_files)

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "tiff"])

# Load image (priority: uploaded -> sample)
image = None
image_name = None
if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image_name = uploaded_file.name
elif selected_sample != "-- none --":
    sample_path = sample_images_dir / selected_sample
    image = cv2.imread(str(sample_path))
    image_name = selected_sample

if image is None:
    st.error(f"Failed to load image. Path attempted: {image_name}")
    st.stop()

# Show original image
st.subheader("Original Image")
st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_column_width=True)

# Build Preprocessor and OCREngine
pre = Preprocessor(
    do_denoise=do_denoise,
    do_clahe=do_clahe,
    do_threshold=do_threshold,
    do_sharpen=do_sharpen,
    do_deskew=do_deskew,
    target_width=int(target_width),
)

ocr = OCREngine(engine=engine)
extractor = TextExtractor(y_threshold=25)

# Run pipeline button
if st.button("Run OCR & Extract ID"):
    with st.spinner("Running preprocessing..."):
        # Many versions of Preprocessor expose either run() or run_pipeline()
        try:
            processed = pre.run(image)
        except Exception:
            processed = pre.run_pipeline(image)

    st.subheader("Processed Image")
    # If processed is single-channel, show with cmap
    if len(processed.shape) == 2:
        st.image(processed, clamp=True, channels="GRAY", use_column_width=True)
    else:
        st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_column_width=True)

    # Run OCR
    with st.spinner("Running OCR..."):
        try:
            ocr_results = ocr.recognize(processed)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            st.stop()

    # Show OCR raw results in a table
    st.subheader("OCR Results (sample)")
    # Prepare a short table for display
    rows = []
    for r in ocr_results:
        rows.append({
            "text": r.get("text", ""),
            "confidence": float(r.get("confidence") or 0),
            "engine": r.get("engine", "")
        })
    st.write("### FULL OCR TEXT:")
    all_text = " ".join([r["text"] for r in ocr_results])
    st.code(all_text)

    # Run extraction
    with st.spinner("Extracting `_1_` token..."):
        result = extractor.extract_target(ocr_results)

    extracted = result.get("extracted_text")
    matched_line = result.get("matched_line")
    all_lines = result.get("all_lines", [])
    raw_groups = result.get("raw_groups", [])

    st.subheader("Extraction Result")
    if result.get("success"):
        st.success(f"Extracted ID: `{extracted}`")
        # Confidence estimate
        conf_est = ocr_results_confidence_for_token(ocr_results, extracted)
        st.write(f"Estimated token confidence: **{conf_est:.2f}**  (average of matched OCR tokens)")

        # Visualize bounding boxes and highlight matched token
        vis = draw_bboxes_on_image(processed if len(processed.shape) > 2 else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR),
                                  ocr_results, highlight_token=extracted)
        st.image(vis, caption="OCR bounding boxes (highlighted token)", use_column_width=True)
        # ---------------- SAVE RESULTS AUTOMATICALLY ----------------
        json_path, img_path = save_result_bundle(
            image_name=image_name,
            extracted_id=extracted,
            extractor_result=result,
            visual_image_rgb=vis,
            confidence=conf_est,
        )

        st.success("Results saved automatically ✅")
        st.write(f"JSON saved to: `{json_path}`")
        st.write(f"Screenshot saved to: `{img_path}`")

    else:
        st.error("No `_1_` token detected.")
        vis = draw_bboxes_on_image(processed if len(processed.shape) > 2 else cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR),
                                  ocr_results, highlight_token=None)
        st.image(vis, caption="OCR bounding boxes", use_column_width=True)
        # ---------------- SAVE FAILED RESULT ----------------
        json_path, img_path = save_result_bundle(
            image_name=image_name,
            extracted_id=None,
            extractor_result=result,
            visual_image_rgb=vis,
            confidence=0.0,
        )

        st.warning("Result saved (extraction failed)")
        st.write(f"JSON saved to: `{json_path}`")
        st.write(f"Screenshot saved to: `{img_path}`")

    # Show matched full line and all lines for debugging
    st.markdown("**Matched full reconstructed line (debug):**")
    st.write(matched_line)

    st.markdown("**All reconstructed lines (debug):**")
    for i, line in enumerate(all_lines, start=1):
        st.write(f"{i}. {line}")

    # Provide JSON download of results
    out = {
        "image_name": image_name,
        "extracted_text": extracted,
        "matched_line": matched_line,
        "all_lines": all_lines,
        "raw_groups": raw_groups,
        "ocr_results": ocr_results,
    }
    out_json = json.dumps(out, default=str, indent=2)

    st.download_button("Download JSON result", data=out_json, file_name=f"{Path(image_name).stem}_ocr_result.json", mime="application/json")

    # Save to results/ folder optionally
    # if st.checkbox("Save result to results/ folder", value=False):
    #     results_dir = Path("results")
    #     results_dir.mkdir(parents=True, exist_ok=True)
    #     save_path = results_dir / f"{Path(image_name).stem}_ocr_result.json"
    #     with open(save_path, "w", encoding="utf-8") as f:
    #         f.write(out_json)
    #     st.write(f"Saved: {save_path}")

# Footer
st.markdown("---")
st.caption("Built for the AI/ML Waybill assessment — extracts the line containing `_1_`. Use easyocr for better results on noisy labels.")
