"""Flask application for Ansel Photo Album."""

import threading

from flask import Flask, jsonify, redirect, render_template, request, send_file, session, url_for

import config
from dropbox_client import dropbox_client
import photo_service
import thumbnail_service
import quality_analyzer
from quality_analyzer import BurstDetector, ReferencePhotoLearner
from theme_classifier import get_theme_classifier

app = Flask(__name__)
app.secret_key = "ansel-photo-album-secret-key-change-in-production"


# --- Pages ---


@app.route("/")
def index():
    """Main UI page."""
    return render_template("index.html")


# --- OAuth Routes ---


@app.route("/oauth/start")
def oauth_start():
    """Start Dropbox OAuth flow."""
    try:
        auth_url = dropbox_client.get_auth_url()
        # Store CSRF token in session
        session["dropbox_csrf"] = dropbox_client._oauth_flow.session.get(
            "dropbox-auth-csrf-token"
        )
        session.modified = True  # Ensure session cookie is set before redirect
        return redirect(auth_url)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/oauth/callback")
def oauth_callback():
    """Handle Dropbox OAuth callback."""
    csrf_token = session.get("dropbox_csrf")
    if not csrf_token:
        return jsonify({"error": "Missing CSRF token"}), 400

    try:
        dropbox_client.complete_auth(request.args, csrf_token)
        return redirect(url_for("index"))
    except Exception as e:
        return jsonify({"error": str(e)}), 400


# --- API Routes ---


@app.route("/api/auth/status")
def auth_status():
    """Check authentication status."""
    is_auth = dropbox_client.is_authenticated()
    account = dropbox_client.get_account_info() if is_auth else None

    # Check if credentials are configured
    app_key, app_secret = config.get_dropbox_credentials()
    has_credentials = bool(app_key and app_secret)

    return jsonify({
        "authenticated": is_auth,
        "has_credentials": has_credentials,
        "account": account,
    })


@app.route("/api/settings", methods=["GET", "POST"])
def settings():
    """Get or update settings."""
    if request.method == "GET":
        app_key, app_secret = config.get_dropbox_credentials()
        return jsonify({
            "dropbox_app_key": app_key or "",
            "dropbox_app_secret": "***" if app_secret else "",
        })

    # POST - update settings
    data = request.json
    if "dropbox_app_key" in data and "dropbox_app_secret" in data:
        config.set_dropbox_credentials(
            data["dropbox_app_key"],
            data["dropbox_app_secret"]
        )

    return jsonify({"success": True})


@app.route("/api/scan", methods=["POST"])
def scan_photos():
    """Scan Dropbox for photos."""
    if not dropbox_client.is_authenticated():
        return jsonify({"error": "Not authenticated"}), 401

    # Get paths from request or use defaults
    data = request.json or {}
    paths = data.get("paths", ["/Camera Uploads", "/Photos"])

    try:
        for i, path in enumerate(paths):
            print(f"\n=== Scanning: {path} ===")
            try:
                # Clear index on first path, merge on subsequent
                index = photo_service.scan_photos(path=path, merge=(i > 0))
            except Exception as e:
                print(f"  Skipping {path}: {e}")

        # Load final merged index
        index = config.load_photo_index()
        return jsonify({
            "success": True,
            "photo_count": len(index.get("photos", {})),
            "year_count": len(index.get("years", {})),
            "paths_scanned": paths,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/folders")
def list_folders():
    """List top-level Dropbox folders."""
    if not dropbox_client.is_authenticated():
        return jsonify({"error": "Not authenticated"}), 401

    try:
        folders = dropbox_client.list_folders()
        return jsonify({"folders": folders})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/years")
def get_years():
    """List all years with photo counts."""
    years = photo_service.get_years()
    return jsonify({"years": years})


@app.route("/api/themes", methods=["GET"])
def get_themes():
    """Get all themes with photo counts."""
    all_themes = config.get_all_themes()
    counts = photo_service.get_theme_counts()

    # Build theme list with counts
    themes = []
    for theme in all_themes:
        themes.append({
            "name": theme,
            "count": counts.get(theme, 0),
            "is_default": theme in config.DEFAULT_THEMES
        })

    # Add unthemed count
    unthemed_count = counts.get("unthemed", 0)

    return jsonify({
        "themes": themes,
        "unthemed_count": unthemed_count
    })


@app.route("/api/themes", methods=["POST"])
def create_theme():
    """Create a custom theme."""
    data = request.json or {}
    theme_name = data.get("name", "").strip()

    if not theme_name:
        return jsonify({"error": "Theme name is required"}), 400

    if config.add_custom_theme(theme_name):
        return jsonify({"success": True, "name": theme_name})
    else:
        return jsonify({"error": "Theme already exists"}), 409


@app.route("/api/themes/<theme_name>", methods=["DELETE"])
def delete_theme(theme_name):
    """Delete a custom theme."""
    if theme_name in config.DEFAULT_THEMES:
        return jsonify({"error": "Cannot delete default themes"}), 400

    if config.remove_custom_theme(theme_name):
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Theme not found or is a default theme"}), 404


@app.route("/api/photos/<photo_id>/themes", methods=["PUT"])
def set_photo_themes(photo_id):
    """Set themes for a single photo."""
    data = request.json or {}
    themes = data.get("themes", [])

    if photo_service.set_photo_themes(photo_id, themes):
        return jsonify({"success": True})
    else:
        return jsonify({"error": "Photo not found"}), 404


@app.route("/api/photos/bulk/themes", methods=["PUT"])
def bulk_set_themes():
    """Set themes for multiple photos."""
    data = request.json or {}
    photo_ids = data.get("photo_ids", [])
    themes = data.get("themes", [])
    mode = data.get("mode", "set")  # "set", "add", or "remove"

    if not photo_ids:
        return jsonify({"error": "No photos specified"}), 400

    updated = photo_service.bulk_set_themes(photo_ids, themes, mode=mode)
    return jsonify({"success": True, "updated": updated})


@app.route("/api/photos/<photo_id>/checked", methods=["PUT"])
def set_photo_checked(photo_id):
    """Set checked state for a single photo (for batch processing)."""
    data = request.json or {}
    checked = data.get("checked", False)

    if photo_service.set_photo_checked(photo_id, checked):
        return jsonify({"success": True, "checked": checked})
    else:
        return jsonify({"error": "Photo not found"}), 404


@app.route("/api/photos/<int:year>")
def get_photos(year):
    """Get all photo metadata for a year, optionally filtered by theme and/or quality."""
    theme = request.args.get("theme")
    quality_filter = request.args.get("quality")  # "good" or "burst_best"
    photos = photo_service.get_photos_for_year(year, theme=theme, quality_filter=quality_filter)
    return jsonify({"photos": photos, "count": len(photos)})


@app.route("/api/thumbnail/<photo_id>")
def get_thumbnail(photo_id):
    """Serve cached thumbnail (generates if missing)."""
    thumbnail_path = thumbnail_service.get_or_create_thumbnail(photo_id)

    if thumbnail_path and thumbnail_path.exists():
        return send_file(thumbnail_path, mimetype="image/jpeg")

    return jsonify({"error": "Thumbnail not found"}), 404


@app.route("/api/photo/<photo_id>/full")
def get_full_photo_link(photo_id):
    """Get a temporary direct link to the full-size photo on Dropbox."""
    if not dropbox_client.is_authenticated():
        return jsonify({"error": "Not authenticated"}), 401

    photo = photo_service.get_photo_by_id(photo_id)
    if not photo:
        return jsonify({"error": "Photo not found"}), 404

    try:
        link = dropbox_client.get_temporary_link(photo["path"])
        return jsonify({"url": link})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/sync/<int:year>", methods=["POST"])
def start_sync(year):
    """Start bulk sync for a year."""
    if not dropbox_client.is_authenticated():
        return jsonify({"error": "Not authenticated"}), 401

    # Check if already syncing
    status = thumbnail_service.get_sync_status()
    if status["is_running"]:
        return jsonify({"error": "Sync already in progress"}), 409

    # Start sync in background thread
    def run_sync():
        for _ in thumbnail_service.sync_year(year):
            pass  # Progress is tracked internally

    thread = threading.Thread(target=run_sync)
    thread.daemon = True
    thread.start()

    return jsonify({"success": True, "message": f"Started syncing year {year}"})


@app.route("/api/sync/status")
def sync_status():
    """Get current sync progress."""
    return jsonify(thumbnail_service.get_sync_status())


# --- Quality Analysis API ---


# Track quality analysis progress
class QualityAnalysisProgress:
    """Track quality analysis progress."""

    def __init__(self):
        self.total = 0
        self.completed = 0
        self.current_file = ""
        self.is_running = False
        self.error = None

    def to_dict(self):
        return {
            "total": self.total,
            "completed": self.completed,
            "current_file": self.current_file,
            "is_running": self.is_running,
            "error": self.error,
            "percent": int((self.completed / self.total * 100) if self.total > 0 else 0),
        }


quality_progress = QualityAnalysisProgress()


@app.route("/api/quality/status")
def quality_status():
    """Get quality analysis progress and statistics."""
    year = request.args.get("year", type=int)
    stats = photo_service.get_quality_stats(year)
    return jsonify({
        "progress": quality_progress.to_dict(),
        "stats": stats,
    })


@app.route("/api/quality/analyze/<int:year>", methods=["POST"])
def analyze_year_quality(year):
    """Analyze quality for all photos in a year that have thumbnails."""
    global quality_progress

    if quality_progress.is_running:
        return jsonify({"error": "Analysis already in progress"}), 409

    data = request.get_json(silent=True) or {}
    force = data.get("force", False)

    def run_analysis():
        global quality_progress

        photos = photo_service.get_photos_for_year(year)
        photos_with_thumbnails = [p for p in photos if p.get("has_thumbnail")]

        if not force:
            photos_to_analyze = [
                p for p in photos_with_thumbnails
                if not p.get("quality", {}).get("analyzed_at")
            ]
        else:
            photos_to_analyze = photos_with_thumbnails

        quality_progress.total = len(photos_to_analyze)
        quality_progress.completed = 0
        quality_progress.is_running = True
        quality_progress.error = None

        try:
            scorer = quality_analyzer.get_quality_scorer()

            for photo in photos_to_analyze:
                quality_progress.current_file = photo["name"]

                try:
                    thumbnail_path = config.THUMBNAILS_DIR / f"{photo['id']}.jpg"
                    if thumbnail_path.exists():
                        with open(thumbnail_path, "rb") as f:
                            image_bytes = f.read()

                        quality_data = scorer.analyze_photo(image_bytes, photo["id"])
                        photo_service.update_photo_quality(photo["id"], quality_data)

                        # Auto-check good photos for batch processing
                        if quality_data.get("is_good_photo"):
                            photo_service.set_photo_checked(photo["id"], True)

                except Exception as e:
                    print(f"Error analyzing {photo['name']}: {e}")

                quality_progress.completed += 1

            # Detect bursts after analysis
            all_photos = photo_service.get_photos_for_year(year)
            detector = BurstDetector()
            burst_groups = detector.detect_bursts(all_photos)
            if burst_groups:
                photo_service.update_burst_info(burst_groups)

        except Exception as e:
            quality_progress.error = str(e)
        finally:
            quality_progress.is_running = False
            quality_progress.current_file = ""

    thread = threading.Thread(target=run_analysis)
    thread.daemon = True
    thread.start()

    return jsonify({
        "success": True,
        "message": f"Started quality analysis for year {year}",
    })


@app.route("/api/quality/calibrate", methods=["POST"])
def calibrate_quality():
    """Calibrate quality thresholds from reference photos."""
    data = request.get_json(silent=True) or {}
    reference_folder = data.get("reference_folder")

    if not reference_folder:
        return jsonify({"error": "reference_folder is required"}), 400

    try:
        learner = ReferencePhotoLearner()
        result = learner.calibrate_from_folder(reference_folder)
        learner.save_calibration(reference_folder, result["thresholds"])
        learner.close()

        return jsonify({
            "success": True,
            "thresholds": result["thresholds"],
            "stats": result["stats"],
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Theme Classification API ---


@app.route("/api/themes/classify/<int:year>", methods=["POST"])
def classify_year_themes(year):
    """Run theme classification on all thumbnails for a year."""
    classifier = get_theme_classifier()

    if not classifier.is_available:
        return jsonify({
            "error": "Theme classifier not available. Train it first or install dependencies."
        }), 400

    def run_classification():
        photos = photo_service.get_photos_for_year(year)
        photos_with_thumbnails = [p for p in photos if p.get("has_thumbnail")]
        classified = 0

        for photo in photos_with_thumbnails:
            try:
                thumbnail_path = config.THUMBNAILS_DIR / f"{photo['id']}.jpg"
                if not thumbnail_path.exists():
                    continue

                with open(thumbnail_path, "rb") as f:
                    image_bytes = f.read()

                result = classifier.predict(image_bytes)
                if result.get("predicted_themes"):
                    photo_service.add_predicted_themes(photo["id"], result["predicted_themes"])
                    classified += 1

            except Exception as e:
                print(f"Error classifying {photo.get('name', photo['id'])}: {e}")

        print(f"Theme classification complete for {year}: {classified}/{len(photos_with_thumbnails)} classified")

    thread = threading.Thread(target=run_classification)
    thread.daemon = True
    thread.start()

    return jsonify({
        "success": True,
        "message": f"Started theme classification for year {year}",
    })


@app.route("/api/themes/classifier/status")
def theme_classifier_status():
    """Get theme classifier status and metadata."""
    classifier = get_theme_classifier()
    metadata = classifier.get_metadata() if classifier.is_available else None

    return jsonify({
        "available": classifier.is_available,
        "metadata": metadata,
    })


# --- Download API ---


# Track download progress
class DownloadProgress:
    """Track download progress."""

    def __init__(self):
        self.total = 0
        self.completed = 0
        self.current_file = ""
        self.is_running = False
        self.error = None
        self.download_path = ""

    def to_dict(self):
        return {
            "total": self.total,
            "completed": self.completed,
            "current_file": self.current_file,
            "is_running": self.is_running,
            "error": self.error,
            "download_path": self.download_path,
            "percent": int((self.completed / self.total * 100) if self.total > 0 else 0),
        }


download_progress = DownloadProgress()


@app.route("/api/download/status")
def download_status():
    """Get download progress."""
    return jsonify(download_progress.to_dict())


@app.route("/api/download/<int:year>", methods=["POST"])
def download_checked_photos(year):
    """Download all checked photos for a year, organized by theme."""
    global download_progress

    if download_progress.is_running:
        return jsonify({"error": "Download already in progress"}), 409

    if not dropbox_client.is_authenticated():
        return jsonify({"error": "Not authenticated"}), 401

    def run_download():
        global download_progress
        import os
        from pathlib import Path

        # Get all checked photos for this year
        photos = photo_service.get_photos_for_year(year)
        checked_photos = [p for p in photos if p.get("checked", False)]

        download_progress.total = 0
        download_progress.completed = 0
        download_progress.is_running = True
        download_progress.error = None

        # Count total downloads (photos can be in multiple themes)
        for photo in checked_photos:
            themes = photo.get("themes", [])
            if themes:
                download_progress.total += len(themes)
            else:
                download_progress.total += 1

        try:
            # Base download directory
            base_dir = Path("/download")
            download_progress.download_path = str(base_dir / str(year))

            # Download each photo
            for photo in checked_photos:
                themes = photo.get("themes", [])
                if not themes:
                    themes = ["unthemed"]

                for theme in themes:
                    download_progress.current_file = f"{theme}/{photo['name']}"

                    try:
                        # Create directory structure: /download/[year]/[theme]/
                        theme_dir = base_dir / str(year) / theme
                        theme_dir.mkdir(parents=True, exist_ok=True)

                        # Download file from Dropbox
                        file_path = theme_dir / photo["name"]

                        # Skip if already exists
                        if not file_path.exists():
                            file_bytes = dropbox_client.download_file(photo["path"])
                            with open(file_path, "wb") as f:
                                f.write(file_bytes)

                    except Exception as e:
                        print(f"Error downloading {photo['name']} to {theme}: {e}")

                    download_progress.completed += 1

        except Exception as e:
            download_progress.error = str(e)
            print(f"Download error: {e}")
        finally:
            download_progress.is_running = False
            download_progress.current_file = ""

    thread = threading.Thread(target=run_download)
    thread.daemon = True
    thread.start()

    return jsonify({
        "success": True,
        "message": f"Started downloading checked photos for year {year}",
    })


if __name__ == "__main__":
    print(f"Starting Ansel Photo Album at http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"Cache directory: {config.ANSEL_DIR}")
    app.run(host=config.FLASK_HOST, port=config.FLASK_PORT, debug=True)
