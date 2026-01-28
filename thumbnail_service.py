"""Thumbnail generation and caching service."""

import io
from pathlib import Path

from PIL import Image

# Register HEIC support
try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

import config
from dropbox_client import dropbox_client
import photo_service
import quality_analyzer
from quality_analyzer import BurstDetector
from theme_classifier import classify_photo


def get_thumbnail_path(photo_id):
    """Get the filesystem path for a cached thumbnail."""
    return config.THUMBNAILS_DIR / f"{photo_id}.jpg"


def has_cached_thumbnail(photo_id):
    """Check if a thumbnail is already cached."""
    return get_thumbnail_path(photo_id).exists()


def generate_thumbnail(image_bytes, photo_id):
    """
    Generate and cache a thumbnail from image bytes.

    Args:
        image_bytes: Raw image file bytes
        photo_id: Photo content hash for cache key

    Returns:
        Path to cached thumbnail
    """
    # Open image
    img = Image.open(io.BytesIO(image_bytes))

    # Handle EXIF orientation
    try:
        exif = img._getexif()
        if exif:
            orientation_key = 274  # EXIF orientation tag
            if orientation_key in exif:
                orientation = exif[orientation_key]
                rotations = {
                    3: 180,
                    6: 270,
                    8: 90,
                }
                if orientation in rotations:
                    img = img.rotate(rotations[orientation], expand=True)
    except (AttributeError, KeyError, TypeError):
        pass

    # Convert to RGB if necessary (for PNG with alpha, HEIC, etc.)
    if img.mode in ("RGBA", "P", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "P":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
        img = background
    elif img.mode != "RGB":
        img = img.convert("RGB")

    # Generate thumbnail maintaining aspect ratio
    img.thumbnail(config.THUMBNAIL_SIZE, Image.Resampling.LANCZOS)

    # Save to cache
    thumbnail_path = get_thumbnail_path(photo_id)
    img.save(thumbnail_path, "JPEG", quality=config.THUMBNAIL_QUALITY)

    return thumbnail_path


def get_or_create_thumbnail(photo_id):
    """
    Get a thumbnail, generating it if needed.

    Args:
        photo_id: Photo content hash

    Returns:
        Path to thumbnail file, or None if photo not found
    """
    # Check cache first
    thumbnail_path = get_thumbnail_path(photo_id)
    if thumbnail_path.exists():
        return thumbnail_path

    # Get photo metadata
    photo = photo_service.get_photo_by_id(photo_id)
    if not photo:
        return None

    # Download from Dropbox
    try:
        _, image_bytes = dropbox_client.download_file(photo["path"])
    except Exception as e:
        print(f"Error downloading {photo['path']}: {e}")
        return None

    # Extract EXIF date and update year if different
    exif_year = photo_service.get_year_from_photo(None, image_bytes)
    if exif_year != photo.get("year"):
        photo_service.update_photo_year(photo_id, exif_year)

    # Generate thumbnail
    try:
        return generate_thumbnail(image_bytes, photo_id)
    except Exception as e:
        print(f"Error generating thumbnail for {photo['path']}: {e}")
        return None


class SyncProgress:
    """Track sync progress for a year."""

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


# Global sync progress tracker
sync_progress = SyncProgress()


def sync_year(year):
    """
    Sync all photos for a year (download and generate thumbnails).

    Args:
        year: Year to sync

    Yields:
        Progress updates as dicts
    """
    global sync_progress

    photos = photo_service.get_photos_for_year(year)

    # Filter to only photos without thumbnails
    photos_to_sync = [p for p in photos if not has_cached_thumbnail(p["id"])]

    sync_progress.total = len(photos_to_sync)
    sync_progress.completed = 0
    sync_progress.is_running = True
    sync_progress.error = None

    synced_photo_ids = []

    try:
        for photo in photos_to_sync:
            sync_progress.current_file = photo["name"]

            # Download and generate thumbnail
            try:
                _, image_bytes = dropbox_client.download_file(photo["path"])

                # Extract EXIF date and update year if needed
                exif_date = photo_service.extract_exif_date(image_bytes)
                if exif_date:
                    photo_service.update_photo_date_taken(photo["id"], exif_date)
                    if exif_date.year != photo.get("year"):
                        photo_service.update_photo_year(photo["id"], exif_date.year)

                # Generate thumbnail
                generate_thumbnail(image_bytes, photo["id"])

                # Analyze photo quality
                quality_data = quality_analyzer.analyze_photo(image_bytes, photo["id"])
                photo_service.update_photo_quality(photo["id"], quality_data)

                # Classify theme (with date for temporal features)
                theme_result = classify_photo(image_bytes, photo["id"], date_taken=exif_date)
                if theme_result.get("predicted_themes"):
                    photo_service.add_predicted_themes(photo["id"], theme_result["predicted_themes"])

                # Mark as synced
                photo_service.mark_photo_synced(photo["id"])
                synced_photo_ids.append(photo["id"])

            except Exception as e:
                print(f"Error syncing {photo['path']}: {e}")

            sync_progress.completed += 1
            yield sync_progress.to_dict()

        # After sync completes, detect bursts for the year
        if synced_photo_ids:
            all_photos = photo_service.get_photos_for_year(year)
            detector = BurstDetector()
            burst_groups = detector.detect_bursts(all_photos)
            if burst_groups:
                photo_service.update_burst_info(burst_groups)

    finally:
        sync_progress.is_running = False
        sync_progress.current_file = ""


def get_sync_status():
    """Get current sync progress."""
    return sync_progress.to_dict()
