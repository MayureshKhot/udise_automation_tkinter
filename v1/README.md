# UDISE OCR Tkinter Tool

This tool reads UDISE code digits from images listed in an Excel sheet and writes the result back to Excel.

It is made for normal users too, not only developers.

## What this app does
- Opens image links/paths from your Excel file
- Finds the UDISE digit area in the image
- Reads each digit using either a local trained model or an external vision API model
- Saves the final code into your output column
- Shows preview before full batch run

## Quick setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install tensorflow
python3 udise_ocr_tkinter_app.py
```

## Simple usage (recommended flow)
1. Select your input file in `Excel`.
2. Choose where to save result in `Output Excel`.
3. Choose `Framework`:
   - `keras` / `torch`: use local file from `Model`
   - `groq`: enter `API Model`, `API Key` (and optional endpoint)
4. Click `Load Model`.
5. Go to **Calibration** tab.
6. Enter a row number in `Sample Row` and click `Load Sample Row`.
7. Mark the digit area (use `Pick ROI` for one full strip, or `Pick Grid Start` for separate digit boxes).
8. Click `Refresh Preview` and check prediction.
9. If needed, adjust settings in `Preprocess Controls`.
10. Click `Save ROI` so you can reuse the same area next time.
11. Click `Run Batch` to process all rows.

## Main tabs
- `Calibration`: test one image and adjust settings
- `Production`: control full batch run behavior

## Buttons and controls (plain language)

### Top section
- `Excel`: your source `.xlsx` file
- `Output Excel`: file where results will be saved
- `Model`: OCR model file (`.keras`, `.h5`, `.pt`, `.pth`)
- `Model Config`: optional model settings JSON
- `API Model`: external model name (example: `meta-llama/llama-4-scout-17b-16e-instruct`)
- `API Key`: external API key (masked in UI)
- `API Endpoint`: external chat completions endpoint
- `Framework`: choose `keras`, `torch`, or `groq`

### Calibration tab - ROI & Sample
- `Load ROI`: load saved ROI/grid settings from file
- `Save ROI`: save current ROI/grid settings
- `Single ROI`: use one big rectangle for all digits
- `Grid ROI (11 boxes)`: use one box per digit
- `Fine-tune Grid`: adjust each digit box position/size manually
- `Apply ROI`: apply ROI typed as `x,y,w,h`
- `Pick ROI`: draw ROI using mouse on sample image
- `Pick Grid Start`: draw first digit box; app creates full grid
- `Load Sample Row`: load image from selected sample row
- `Find Next Pending`: jump to next row where output is empty

### Calibration tab - Preprocess Controls
- `Digit Count`: expected number of digits (usually 11)
- `Image Size`: model input size (usually auto-adjusted)
- `Invert`: switches black/white style for better digit visibility
- `Tight Crop`: trims extra blank space around digit
- `Blur Kernel`: smooths noise
- `Threshold`: digit separation mode (`otsu` or `adaptive`)
- `Adaptive Block`, `Adaptive C`: fine controls for adaptive threshold
- `BBox Padding`: extra margin around cropped digit
- `Remove Vertical Grid Lines`: tries to remove thin vertical table lines
- `Line Min H Ratio`, `Line Max Width`, `Edge Margin`: controls for line removal

### Calibration tab - action buttons
- `Load Model`: loads selected model
- `Refresh Preview`: reruns prediction on sample with current settings
- `Run Batch`: starts full processing

### Right panel
- `Image Preview`: full image + ROI view
- `Preprocessed Digits`: final digit tiles used for prediction
- `Global Shift` (`↑ ↓ ← →`): moves ROI/grid for quick alignment
- `Step`: how many pixels to move each click
- `Logs`: live progress and errors

### Production tab (batch behavior)
- `Start Row`: first Excel row to process
- `Max Rows (0=all)`: row limit; `0` means till end
- `Timeout`, `Retries`, `Backoff`: controls for loading images
- `Save Every`: autosave frequency during batch
- `Overwrite Existing`: replace values already present in output column
- `Manual Review`: pauses when confidence is low
- `Threshold`: confidence cutoff for manual review

## Output files
- Result Excel: usually `<original_name>_with_udise.xlsx`
- Failure log: `udise_failures_gui.csv` (rows that had errors)

## Common tips
- If preview says model not loaded, click `Load Model` first.
- If prediction is wrong, first check ROI placement, then tweak preprocess controls.
- Use `Save ROI` after calibration so future runs are faster.
- Keep `Digit Count` same as your expected UDISE code length.
