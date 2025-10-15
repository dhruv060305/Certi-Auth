# Certificate Model Trainer

Lightweight tool to extract image/text features from certificate files and train classifiers to detect authentic vs fake certificates.

## Features
- Extract visual, texture, frequency, geometric and OCR-derived features.
- Train multiple models (Random Forest, Gradient Boosting, SVM) and compare performance.
- Save trained models, scaler, feature dumps and reports (plots + text).
- Predict/classify unlabeled certificates.

## Project structure
- certificate_data/
  - authentic/      — put authentic certificate files (PDF, JPG, PNG, etc.)
  - fake/           — put fake certificate files
  - unlabeled/      — files to classify after training
  - models/         — saved model and scaler files
  - features/       — saved extracted features (pickle)
  - reports/        — training reports and plots

## Prerequisites
- Python 3.8+
- Recommended: create and use a virtual environment.

Optional (for PDF & OCR):
- PyMuPDF (fitz) — to load PDFs
- pytesseract + Tesseract OCR binary — for OCR features

On Windows, install Tesseract from: https://github.com/tesseract-ocr/tesseract and add its installation folder to PATH.

## Installation
Install Python dependencies:
```powershell
python -m pip install -r requirements.txt
```

Install Tesseract (Windows): download installer, run it, then add `C:\Program Files\Tesseract-OCR` (or your install path) to PATH.

## Usage
From project root:

- Extract features and train:
```powershell
python trainer.py --train
```

- Force re-extraction of features and retrain:
```powershell
python trainer.py --train --reprocess
```

- Hyperparameter tuning (Random Forest grid search):
```powershell
python trainer.py --tune
```

- Classify unlabeled files:
```powershell
python trainer.py --predict
```

- Specify custom data directory:
```powershell
python trainer.py --train --data-dir "C:\path\to\data"
```

## Outputs
- models/ — saved best model and tuned model files
- features/training_features.pickle — cached features for faster iteration
- reports/ — detailed training report (.txt), plots (.png), and classification results (.json)

## Tips
- Ensure dataset is balanced or use sampling techniques to avoid biased models.
- If OCR is unavailable, OCR-derived features will be zero and the rest will still work.
- Increase dataset size and augment images for better generalization.
- Inspect `reports/` after training for confusion matrix, feature importances and metrics.

## Troubleshooting
- "Could not load image": ensure OpenCV supports the file format; convert PDFs or images if necessary.
- Tesseract OCR not found: confirm Tesseract binary is installed and in PATH.
- Long training times: reduce `n_estimators` or sample dataset for experimentation.


