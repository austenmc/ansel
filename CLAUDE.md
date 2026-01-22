# Ansel Photo Album

A local photo album web app that syncs from Dropbox, generates thumbnails, and provides AI-based quality classification.

## Architecture

- **Backend**: Flask (Python 3.14), serves API + static files
- **Frontend**: Vanilla JS/HTML/CSS single-page app
- **Storage**: Dropbox API for photo source, local filesystem for thumbnails/index
- **ML**: OpenCV for quality analysis, MediaPipe for face detection (optional)

## Key Files

| File | Purpose |
|------|---------|
| `app.py` | Flask routes and API endpoints |
| `config.py` | Config, paths, photo index load/save (thread-safe) |
| `dropbox_client.py` | Dropbox API wrapper |
| `photo_service.py` | Photo indexing, EXIF, year grouping, themes, quality filtering |
| `thumbnail_service.py` | Thumbnail generation, sync orchestration |
| `quality_analyzer.py` | ML quality scoring (sharpness, exposure, composition, color, faces) |
| `analyze_existing.py` | CLI script to batch-analyze existing thumbnails |
| `templates/index.html` | Main UI template |
| `static/app.js` | Frontend application logic |
| `static/style.css` | Styles (dark theme) |

## Data Storage

All persistent data lives in `~/.ansel/`:
- `config.json` - Dropbox credentials
- `dropbox_token.json` - OAuth tokens
- `photo_index.json` - Photo metadata, themes, quality scores (thread-safe with lock + atomic writes)
- `cache/thumbnails/` - JPEG thumbnail files keyed by content hash

## Running

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py  # Starts on http://localhost:5001
```

## CLI Tools

```bash
# Analyze quality for existing cached thumbnails
python analyze_existing.py [--year YEAR] [--force] [--verbose] [--stats-only]
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
- NumPy types must be converted to native Python types before JSON serialization
- The app uses `request.get_json(silent=True)` for POST endpoints that may not have a body
- Thumbnails are 300x300 JPEG, keyed by Dropbox content hash
