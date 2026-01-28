# Ansel Photo Album

A local photo album web app that syncs from Dropbox, generates thumbnails, and provides AI-based quality and theme classification.

## Architecture

- **Backend**: Flask (Python 3.12), serves API + static files
- **Frontend**: Vanilla JS/HTML/CSS single-page app
- **Storage**: Dropbox API for photo source, local filesystem for thumbnails/index
- **ML**: OpenCV for quality analysis, MobileNetV2 + FaceNet + SVM for theme classification, MediaPipe for face detection (optional)

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Flask routes and API endpoints |
| `config.py` | Config, paths, photo index load/save (thread-safe) |
| `dropbox_client.py` | Dropbox API wrapper |
| `photo_service.py` | Photo indexing, EXIF, year grouping, themes, quality filtering |
| `thumbnail_service.py` | Thumbnail generation, sync orchestration |
| `quality_analyzer.py` | ML quality scoring (sharpness, exposure, composition, color, faces) |
| `theme_classifier.py` | Theme classification: MobileNetV2 features + face embeddings + temporal encoding + SVM |
| `train_themes.py` | CLI script to train the theme classifier from labeled photo folders |
| `evaluate_classifier.py` | CLI script to evaluate classifier accuracy on a held-out test set |
| `diagnose_classifier.py` | CLI script to diagnose classifier performance (confidence distribution, face detection rate) |
| `analyze_existing.py` | CLI script to batch-analyze existing thumbnails (quality + themes) |
| `templates/index.html` | Main UI template |
| `static/app.js` | Frontend application logic |
| `static/style.css` | Styles (dark theme) |

## Data Storage

All persistent data lives in `~/.ansel/`:
- `config.json` - Dropbox credentials
- `dropbox_token.json` - OAuth tokens
- `photo_index.json` - Photo metadata, themes, quality scores, date_taken (thread-safe with lock + atomic writes)
- `cache/thumbnails/` - JPEG thumbnail files keyed by content hash
- `models/theme_classifier.pkl` - Trained SVM classifier + label encoder
- `models/theme_classifier_meta.json` - Training metadata (classes, accuracy, feature dimensions)

## Setup

Requires Python 3.12 (PyTorch does not support Python 3.13+).

```bash
# Create virtual environment with Python 3.12
python3.12 -m venv venv312
source venv312/bin/activate

# Install base dependencies
pip install -r requirements.txt

# Install ML dependencies for theme classification
pip install torch torchvision scikit-learn

# Install face embeddings (optional but recommended)
pip install facenet-pytorch
```

## Running

```bash
source venv312/bin/activate
python app.py  # Starts on http://localhost:5001
```

## Theme Classifier

The theme classifier uses a 1794-dimensional feature vector:
- **MobileNetV2 visual features** (1280-d): Pretrained ImageNet features from the penultimate layer
- **Face embeddings** (512-d): FaceNet VGGFace2 embeddings via MTCNN face detection (zeros if no face or facenet-pytorch not installed)
- **Temporal features** (2-d): Cyclical sin/cos encoding of the month from EXIF date

An SVM (LinearSVC + CalibratedClassifierCV) is trained on these features to predict theme labels. The classifier always assigns its top prediction to each photo (no confidence threshold filtering).

### Training

Organize labeled photos into folders by theme:

```
~/Desktop/2022 Photos/
├── Christmas/
│   ├── IMG_001.jpg
│   └── IMG_002.jpg
├── Brooke/
│   ├── IMG_003.jpg
│   └── ...
├── Liam/
└── Friends/
```

Birthday folders are normalized: `Brooke's BDay` -> `Brooke`.

Train the classifier:

```bash
source venv312/bin/activate

# Train from one or more labeled directories
python train_themes.py ~/Desktop/"2022 Photos" ~/Desktop/"2021 Photos"

# With options
python train_themes.py --min-samples 10 --verbose ~/Desktop/"2022 Photos"
```

The trained model is saved to `~/.ansel/models/`.

### Evaluating

Evaluate classifier accuracy on a held-out test set (labeled photos not used for training):

```bash
# Basic evaluation
python evaluate_classifier.py ~/Desktop/"2020 Photos"/Done/

# Per-image verbose output
python evaluate_classifier.py --verbose ~/Desktop/"2020 Photos"/Done/
```

Reports per-class accuracy, overall precision/recall, and common misclassifications.

### Diagnosing

Diagnose confidence distribution and face detection rates:

```bash
python diagnose_classifier.py ~/Desktop/"2020 Photos"/Done/
```

Reports face detection rate, confidence percentiles, and per-class confidence ranges.

### Retraining

Retrain the classifier after:
- Adding new training data
- Installing/uninstalling facenet-pytorch (changes feature dimensions)
- Adding new theme categories

## CLI Tools

```bash
# Analyze quality for existing cached thumbnails
python analyze_existing.py [--year YEAR] [--force] [--verbose] [--stats-only]

# Analyze quality AND classify themes
python analyze_existing.py --classify-themes [--year YEAR] [--force]

# Train theme classifier
python train_themes.py [--min-samples N] [--verbose] <training_dirs...>

# Evaluate classifier on test set
python evaluate_classifier.py [--verbose] <test_dir>

# Diagnose classifier performance
python diagnose_classifier.py [<test_dir>]
```

## API Endpoints

- `GET /api/auth/status` - Auth check
- `POST /api/scan` - Scan Dropbox folders
- `GET /api/years` - List years with counts
- `GET /api/photos/<year>?theme=X&quality=good|burst_best` - Get photos
- `POST /api/sync/<year>` - Sync thumbnails for year
- `GET /api/sync/status` - Sync progress
- `POST /api/quality/analyze/<year>` - Run quality analysis
- `GET /api/quality/status?year=X` - Analysis progress + stats
- `POST /api/quality/calibrate` - Calibrate from reference folder

## Quality Analysis

Scores 0-100 on four dimensions (all photo types):
- **Sharpness**: Laplacian variance
- **Exposure**: Histogram clipping analysis
- **Composition**: Edge detection + rule of thirds
- **Color**: Saturation + variance

Optional face analysis (requires mediapipe): eye aspect ratio, smile detection.

Burst detection groups photos within 2 seconds and marks the best.

## Important Notes

- `photo_index.json` uses thread locking (`threading.Lock`) and atomic writes to prevent corruption during concurrent access from sync/analysis threads
- MediaPipe is optional; face analysis is skipped if unavailable
- FaceNet is optional; face embeddings are zeros if facenet-pytorch is not installed. Retrain the classifier if you change whether it's installed.
- NumPy types must be converted to native Python types before JSON serialization
- The app uses `request.get_json(silent=True)` for POST endpoints that may not have a body
- Thumbnails are 300x300 JPEG, keyed by Dropbox content hash
- `photo_service.py` uses lazy import for `dropbox_client` so that offline tools (training, evaluation) don't require the dropbox package
