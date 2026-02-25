# UDISE OCR Tkinter Tool (v2 Minimal)

This is a simplified UI-focused version of your app.

## Kept features
- OMR image preview (full image + ROI preview)
- ROI selection from image
- Move ROI with `Up / Down / Left / Right`
- Zoomable ROI picker (`Zoom +`, `Zoom -`, `100%`, `Fit`, mouse wheel)
- Model loading (`keras`/`torch`) and prediction
- Digit previews with predicted value and confidence
- Excel import/export for row-wise workflow
- URL queue scanning from Excel (`Scan URLs`)
- One-by-one navigation (`Previous` / `Next`)
- Single prediction write-back (`Predict + Write`)
- Bulk processing for all scanned URLs (`Run All URLs`)
- Dynamic gridline suppression in preprocessing (removes box lines, especially vertical separators)

## Removed from v1
- Full batch processing and production tab
- Logs tab and manual review workflow
- API/Groq integration
- Grid tuning/fine offset editors
- Advanced preprocess control panel

## Quick start
```bash
cd v2
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install tensorflow
python3 udise_ocr_tkinter_app.py
```

## Usage flow
1. Select `Excel Input` and optional `Output` Excel.
2. Set `Sheet`, `Src Col` (image URL column), `Out Col` (UDISE result column), and `Start Row`.
3. Click `Scan URLs` to build the row queue.
4. Click `Load Row Image` for current row, or use `Previous`/`Next` to move through rows.
5. Load the model.
6. Select the correct ROI once and fine-adjust with arrow buttons.
7. For single row: click `Predict + Write`.
8. For full sheet utility run: click `Run All URLs` to process all scanned rows and write predictions in output column.
