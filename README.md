Certificate Model Trainer

Place authentic and fake certificates into `certificate_data/authentic` and `certificate_data/fake` respectively.

Common tasks:
- Extract features and train: python trainer.py --train
- Force re-extraction: python trainer.py --train --reprocess
- Hyperparameter tuning: python trainer.py --tune
- Classify unlabeled files: python trainer.py --predict

Notes:
- Install dependencies from `requirements.txt` (prefer a virtual environment).
- For PDF and OCR support install `PyMuPDF` and `pytesseract` and ensure Tesseract binary is installed on your system.
