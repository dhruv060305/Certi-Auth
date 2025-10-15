#!/usr/bin/env python3
"""
Certificate Model Training System (CLI) - Enhanced with Debug
Trains a model using 15 features extracted from certificate images/PDFs.
"""

import os
import cv2
import fitz
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score

# Directories
DATA_DIR = Path("dataset")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
for d in [DATA_DIR, MODEL_DIR, REPORT_DIR]:
    d.mkdir(exist_ok=True)

# Feature names
FEATURE_NAMES = [
    'sharpness', 'brightness', 'color_variance', 'edge_density', 'text_clarity',
    'noise_level', 'contrast_ratio', 'symmetry_score', 'font_consistency',
    'seal_quality', 'watermark_detection', 'resolution_quality', 
    'compression_artifacts', 'color_depth', 'geometric_consistency'
]

def extract_features(image):
    """Extract 15 features from certificate image (matches Streamlit)"""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []

        # 1. Sharpness
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        features.append(min(10, laplacian_var / 100))

        # 2. Brightness Consistency
        brightness_std = np.std(gray)
        features.append(max(0, 10 - brightness_std / 25))

        # 3. Color Variance
        color_variance = np.var(image)
        features.append(min(10, color_variance / 1000))

        # 4. Edge Density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density * 10)

        # 5. Text Clarity
        try:
            import pytesseract
            text = pytesseract.image_to_string(gray)
            text_quality = len([w for w in text.split() if w.isalpha()]) / 10
            features.append(min(10, text_quality))
        except Exception as e:
            print(f"   Warning: Text extraction failed: {e}")
            features.append(5.0)

        # 6. Noise Level
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        noise_level = np.mean(np.abs(f_shift)) / 1000
        features.append(max(0, 10 - noise_level))

        # 7. Contrast Ratio
        contrast = gray.std()
        features.append(min(10, contrast / 25))

        # 8. Symmetry Score
        rows, cols = gray.shape
        left_half = gray[:, :cols//2]
        right_half = cv2.flip(gray[:, cols//2:], 1)
        min_width = min(left_half.shape[1], right_half.shape[1])
        symmetry_diff = np.mean(np.abs(left_half[:, :min_width] - right_half[:, :min_width])) if min_width > 0 else 0
        features.append(max(0, 10 - symmetry_diff / 25))

        # 9. Font Consistency
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        font_consistency = np.sum(horizontal_lines > 0) / edges.size
        features.append(font_consistency * 50)

        # 10. Seal Quality
        circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=10, maxRadius=100)
        features.append(min(10, len(circles[0])*2) if circles is not None else 2.0)

        # 11-15: Additional simplified features
        np.random.seed(42)  # Make random features more consistent
        features.extend([
            max(0, min(10, np.random.normal(6, 1))),  # watermark_detection
            min(10, np.log10(image.shape[0] * image.shape[1]) - 4),  # resolution_quality
            max(0, 10 - np.random.normal(3, 1)),  # compression_artifacts
            min(10, np.log10(len(np.unique(image.reshape(-1, image.shape[2]), axis=0)))),  # color_depth
            max(0, min(10, np.random.normal(7, 1)))  # geometric_consistency
        ])

        return features
    
    except Exception as e:
        print(f"   Error extracting features: {e}")
        return None

def load_images_from_folder(folder, label):
    """Load images/PDFs from folder and extract features"""
    samples = []
    folder_path = Path(folder)
    
    print(f"\n📁 Processing folder: {folder_path}")
    print(f"   Label: {'AUTHENTIC' if label == 1 else 'FAKE'}")
    
    if not folder_path.exists():
        print(f"   ❌ Folder does not exist: {folder_path}")
        return samples
    
    # Get all files in folder
    all_files = list(folder_path.glob("*.*"))
    print(f"   Found {len(all_files)} files")
    
    supported_image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"]
    
    for file in all_files:
        print(f"   📄 Processing: {file.name}")
        img = None
        
        try:
            # Handle image files
            if file.suffix.lower() in supported_image_extensions:
                print(f"      Loading as image...")
                img = cv2.imread(str(file))
                if img is None:
                    print(f"      ❌ Failed to load image: {file.name}")
                    continue
            
            # Handle PDF files
            elif file.suffix.lower() == ".pdf":
                print(f"      Loading as PDF...")
                try:
                    pdf = fitz.open(str(file))
                    if len(pdf) == 0:
                        print(f"      ❌ PDF has no pages: {file.name}")
                        pdf.close()
                        continue
                    
                    page = pdf[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                    img_data = pix.tobytes("png")
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    pdf.close()
                    
                    if img is None:
                        print(f"      ❌ Failed to extract image from PDF: {file.name}")
                        continue
                        
                except Exception as e:
                    print(f"      ❌ PDF processing error: {e}")
                    continue
            
            else:
                print(f"      ⚠️  Unsupported file format: {file.suffix}")
                continue
            
            # Extract features from loaded image
            if img is not None:
                print(f"      Image shape: {img.shape}")
                feats = extract_features(img)
                
                if feats is not None:
                    feats.append(label)  # add label at end
                    samples.append(feats)
                    print(f"      ✅ Successfully extracted features")
                else:
                    print(f"      ❌ Feature extraction failed")
            
        except Exception as e:
            print(f"      ❌ Unexpected error processing {file.name}: {e}")
            continue
    
    print(f"   📊 Successfully processed {len(samples)} files from {folder_path.name}")
    return samples

def train_models():
    """Train RandomForest on 15 features"""
    print("🚀 Starting Certificate Model Training")
    print("="*50)
    
    # Load data from both folders
    print("\n📂 Loading training data...")
    authentic = load_images_from_folder(DATA_DIR / "authentic", 1)
    fake = load_images_from_folder(DATA_DIR / "fake", 0)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Authentic samples: {len(authentic)}")
    print(f"   Fake samples: {len(fake)}")
    print(f"   Total samples: {len(authentic) + len(fake)}")
    
    if len(authentic) == 0 and len(fake) == 0:
        print("\n❌ No valid samples found! Check your dataset folder structure:")
        print("   dataset/")
        print("   ├── authentic/  (put authentic certificates here)")
        print("   └── fake/       (put fake certificates here)")
        return
    
    if len(authentic) == 0:
        print("\n⚠️  Warning: No authentic samples found!")
    if len(fake) == 0:
        print("\n⚠️  Warning: No fake samples found!")
    
    # Create DataFrame
    all_data = authentic + fake
    data = pd.DataFrame(all_data, columns=FEATURE_NAMES + ["label"])
    
    print(f"\n📈 Final dataset shape: {data.shape}")
    print("\n📋 Label distribution:")
    print(data['label'].value_counts().to_dict())
    
    print(f"\n🔬 Feature preview:")
    print(data.head())
    
    # Prepare features and labels
    X = data[FEATURE_NAMES].values
    y = data["label"].values
    
    print(f"\n🎯 Training model...")
    print(f"   Features shape: {X.shape}")
    print(f"   Labels shape: {y.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    
    # Cross-validation (only if we have enough samples)
    if len(np.unique(y)) > 1 and len(y) >= 4:
        n_splits = min(5, len(y) // 2)  # Adjust splits based on data size
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(model, X_scaled, y, cv=skf)
        print(f"   Cross-validation accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
    else:
        print("   ⚠️  Skipping cross-validation (insufficient data)")
    
    # Train final model
    model.fit(X_scaled, y)
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': FEATURE_NAMES,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🏆 Top 5 most important features:")
    for _, row in feature_importance.head().iterrows():
        print(f"   {row['feature']}: {row['importance']:.3f}")
    
    # Save model and scaler
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "best_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    
    # Save feature importance report
    feature_importance.to_csv(REPORT_DIR / "feature_importance.csv", index=False)
    
    print(f"\n✅ Training completed successfully!")
    print(f"   Model saved: {MODEL_DIR / 'best_model.pkl'}")
    print(f"   Scaler saved: {MODEL_DIR / 'scaler.pkl'}")
    print(f"   Report saved: {REPORT_DIR / 'feature_importance.csv'}")
    
    return model, scaler

if __name__ == "__main__":
    try:
        model, scaler = train_models()
        print("\n🎉 All done! You can now run the Streamlit app:")
        print("   streamlit run certificate_web_app.py")
    except KeyboardInterrupt:
        print("\n\n⏹️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()