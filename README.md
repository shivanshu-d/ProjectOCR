# Waybill `_1_` ID Extraction System

This project implements an OCR-based system to extract shipment / order IDs containing the `_1_` pattern from logistics waybill images.

The solution is designed for **three known label layouts** and uses **open-source OCR engines** with a **rule-based extraction strategy** for reliability and explainability.

---

## 📌 Key Features

- Supports multiple waybill layouts (horizontal, vertical, AWB-style)
- Uses EasyOCR and Tesseract OCR (open-source)
- Robust rule-based ID extraction (no training required)
- Ignores barcode artifacts and OCR noise
- Streamlit-based interactive UI
- Automatically saves results and OCR screenshots
- Reproducible evaluation artifacts (`results/` folder)

---

##  Project Structure

ProjectOCR/
│
├── app.py # Streamlit application
├── requirements.txt # Python dependencies
├── README.md # Project documentation
│
├── src/
│ ├── preprocessing.py # Image preprocessing pipeline
│ ├── ocr_engine.py # EasyOCR + Tesseract wrapper
│ ├── text_extraction.py # Rule-based ID extractor
│ └── utils_results.py # Automatic result saving utilities
│
├── datasets/
│ └── ReverseWay Bill/ # Sample waybill images
│
├── results/ # Auto-generated outputs
│ └── <image_name>/
│ ├── result.json
│ └── ocr_visual.png

##  Install dependencies

pip install -r requirements.txt

brew install tesseract      #install TesseractOCR
brew install easyocr        #install EasyOCR

##  Running the application 

streamlit run app.py

The app allows you to:
Upload a waybill image
Apply preprocessing
Run OCR
Extract the _1_ ID
View highlighted OCR results
Automatically save outputs

##  Outputs

results/<image_name>/
├── result.json       # Extracted ID, confidence, metadata
└── ocr_visual.png    # OCR bounding boxes + highlighted ID

##  Extraction Logic

Direct Match
1. Extract IDs matching:

<12+ digits>_1(_suffix)?


2. Numeric Fallback
If _1_ is missing, select the longest valid numeric token (excluding barcodes) and reconstruct:

<number>_1_

##  Evaluation

Metric: Exact Match Accuracy
Confidence: Average OCR confidence of matched tokens
Results and screenshots are saved automatically for reproducibility

## Limitations

Uses only open-source OCR engines
No proprietary APIs
Fully reproducible
Explainable logic

## Compliance

Uses only open-source OCR engines
No proprietary APIs
Fully reproducible
Explainable logic