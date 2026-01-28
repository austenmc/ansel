"""Photo indexing service with EXIF extraction and year grouping."""

import io
from datetime import datetime

from PIL import Image
from PIL.ExifTags import TAGS

import config

# Lazy import to avoid requiring dropbox for offline operations (training, etc.)
dropbox_client = None

def _get_dropbox_client():
    """Lazy-load dropbox client."""
    global dropbox_client
    if dropbox_client is None:
        from dropbox_client import dropbox_client as _client
        dropbox_client = _client
    return dropbox_client


def extract_exif_date(image_bytes):
    """
    Extract date taken from EXIF data.

    Args:
        image_bytes: Raw image file bytes

    Returns:
        datetime object or None if no EXIF date found
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        exif_data = img._getexif()

        if not exif_data:
            return None

        # Look for DateTimeOriginal (36867) or DateTime (306)
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "DateTimeOriginal":
                # Format: "2023:01:15 14:30:00"
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
            elif tag_name == "DateTime" and tag_id == 306:
                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")

        return None
    except Exception:
        return None


def get_year_from_photo(entry, image_bytes=None):
    """
    Determine the year a photo was taken.

    Args:
        entry: Dropbox FileMetadata entry
        image_bytes: Optional image bytes for EXIF extraction

    Returns:
        Year as integer
    """
    # Try EXIF date first if we have the image bytes
    if image_bytes:
        exif_date = extract_exif_date(image_bytes)
        if exif_date:
            return exif_date.year

    # Fall back to Dropbox client_modified timestamp
    if entry.client_modified:
        return entry.client_modified.year

    # Last resort: server modified
    if entry.server_modified:
        return entry.server_modified.year

    return datetime.now().year


def is_image_file(path):
    """Check if a file path is a supported image format."""
    return path.lower().endswith(tuple(config.IMAGE_EXTENSIONS))


def scan_photos(progress_callback=None, path="", merge=True):
    """
    Scan Dropbox for all photos and build an index.

    Args:
        progress_callback: Optional callback(scanned_count) for progress updates
        path: Dropbox path to scan (empty string for root, or e.g. "/Photos")
        merge: If True, merge with existing index; if False, start fresh

    Returns:
        Dictionary with photo index
    """
    if merge:
        index = config.load_photo_index()
        if not index.get("photos"):
            index = {"photos": {}, "years": {}}
    else:
        index = {"photos": {}, "years": {}}
    scanned = 0
    files_seen = 0

    print(f"Starting scan of Dropbox path: '{path or '/'}'")

    for entry in _get_dropbox_client().list_folder_recursive(path):
        files_seen += 1
        if files_seen % 500 == 0:
            print(f"  Checked {files_seen} files, found {scanned} photos...")

        if not is_image_file(entry.path_lower):
            continue

        print(f"  Found: {entry.path_display}")

        # Use content hash as unique ID
        photo_id = entry.content_hash

        # Get year from Dropbox metadata (without downloading for now)
        year = entry.client_modified.year if entry.client_modified else datetime.now().year

        # Store photo metadata
        index["photos"][photo_id] = {
            "id": photo_id,
            "path": entry.path_display,
            "path_lower": entry.path_lower,
            "name": entry.name,
            "size": entry.size,
            "year": year,
            "content_hash": entry.content_hash,
            "client_modified": entry.client_modified.isoformat() if entry.client_modified else None,
        }

        # Update year counts
        year_str = str(year)
        if year_str not in index["years"]:
            index["years"][year_str] = {"count": 0, "synced": 0}
        index["years"][year_str]["count"] += 1

        scanned += 1
        if progress_callback and scanned % 100 == 0:
            progress_callback(scanned)

    # Final callback
    if progress_callback:
        progress_callback(scanned)

    print(f"Scan complete: {scanned} photos found in {files_seen} files")

    # Save index
    config.save_photo_index(index)
    return index


def get_years():
    """
    Get list of years with photo counts.

    Returns:
        List of dicts with year, count, and synced count
    """
    index = config.load_photo_index()
    years = []

    for year, data in sorted(index.get("years", {}).items(), reverse=True):
        years.append({
            "year": int(year),
            "count": data["count"],
            "synced": data.get("synced", 0),
        })

    return years


def get_photos_for_year(year, theme=None, quality_filter=None):
    """
    Get all photos for a specific year, optionally filtered by theme and/or quality.

    Args:
        year: Year as integer or string
        theme: Optional theme filter. Use "unthemed" for photos with no themes.
        quality_filter: Optional quality filter:
            - "good": Only photos with is_good_photo=True
            - "burst_best": Only best photos from each burst (is_burst_best=True)

    Returns:
        List of photo metadata dicts
    """
    index = config.load_photo_index()
    year_str = str(year)

    photos = []
    for photo_id, photo in index.get("photos", {}).items():
        if str(photo.get("year")) == year_str:
            photo_themes = photo.get("themes", [])

            # Filter by theme if specified
            if theme:
                if theme.lower() == "unthemed":
                    # Only include photos with no themes
                    if photo_themes:
                        continue
                else:
                    # Only include photos with the specified theme
                    if theme not in photo_themes:
                        continue

            # Filter by quality if specified
            if quality_filter:
                quality = photo.get("quality", {})
                if quality_filter == "good":
                    if not quality.get("is_good_photo", False):
                        continue
                elif quality_filter == "burst_best":
                    # Include if not in a burst OR is the best of its burst
                    burst_group_id = quality.get("burst_group_id")
                    if burst_group_id and not quality.get("is_burst_best", False):
                        continue

            # Check if thumbnail is cached
            thumbnail_path = config.THUMBNAILS_DIR / f"{photo_id}.jpg"
            photo["has_thumbnail"] = thumbnail_path.exists()
            photos.append(photo)

    # Sort by date, newest first
    photos.sort(key=lambda p: p.get("client_modified", ""), reverse=True)
    return photos


def get_photo_by_id(photo_id):
    """
    Get a single photo's metadata by ID.

    Args:
        photo_id: Photo content hash

    Returns:
        Photo metadata dict or None
    """
    index = config.load_photo_index()
    return index.get("photos", {}).get(photo_id)


def update_photo_year(photo_id, year):
    """
    Update the year for a photo (after EXIF extraction).

    Args:
        photo_id: Photo content hash
        year: New year value
    """
    index = config.load_photo_index()

    if photo_id not in index.get("photos", {}):
        return

    old_year = str(index["photos"][photo_id].get("year"))
    new_year = str(year)

    if old_year == new_year:
        return

    # Update photo
    index["photos"][photo_id]["year"] = year

    # Update year counts
    if old_year in index["years"]:
        index["years"][old_year]["count"] -= 1
        if index["years"][old_year]["count"] <= 0:
            del index["years"][old_year]

    if new_year not in index["years"]:
        index["years"][new_year] = {"count": 0, "synced": 0}
    index["years"][new_year]["count"] += 1

    config.save_photo_index(index)


def mark_photo_synced(photo_id):
    """Mark a photo as synced (thumbnail generated)."""
    index = config.load_photo_index()

    if photo_id not in index.get("photos", {}):
        return

    year_str = str(index["photos"][photo_id].get("year"))
    if year_str in index["years"]:
        index["years"][year_str]["synced"] = index["years"][year_str].get("synced", 0) + 1

    config.save_photo_index(index)


def update_photo_date_taken(photo_id, date_taken):
    """
    Update the date_taken for a photo.

    Args:
        photo_id: Photo content hash
        date_taken: datetime object

    Returns:
        True if successful, False otherwise
    """
    index = config.load_photo_index()

    if photo_id not in index.get("photos", {}):
        return False

    index["photos"][photo_id]["date_taken"] = date_taken.isoformat()
    config.save_photo_index(index)
    return True


def set_photo_themes(photo_id, themes):
    """
    Set themes for a single photo.

    Args:
        photo_id: Photo content hash
        themes: List of theme names

    Returns:
        True if successful, False otherwise
    """
    index = config.load_photo_index()

    if photo_id not in index.get("photos", {}):
        return False

    index["photos"][photo_id]["themes"] = themes
    config.save_photo_index(index)
    return True


def bulk_set_themes(photo_ids, themes, mode="set"):
    """
    Set themes for multiple photos in a single operation.

    Args:
        photo_ids: List of photo content hashes
        themes: List of theme names
        mode: "set" to replace themes, "add" to add themes, "remove" to remove themes

    Returns:
        Number of photos updated
    """
    index = config.load_photo_index()
    updated = 0

    for photo_id in photo_ids:
        if photo_id not in index.get("photos", {}):
            continue

        if mode == "set":
            index["photos"][photo_id]["themes"] = themes
        elif mode == "add":
            existing = set(index["photos"][photo_id].get("themes", []))
            existing.update(themes)
            index["photos"][photo_id]["themes"] = list(existing)
        elif mode == "remove":
            existing = set(index["photos"][photo_id].get("themes", []))
            existing -= set(themes)
            index["photos"][photo_id]["themes"] = list(existing)

        updated += 1

    config.save_photo_index(index)
    return updated


def get_theme_counts():
    """
    Get count of photos for each theme.

    Returns:
        Dict mapping theme names to counts, plus "unthemed" count
    """
    index = config.load_photo_index()
    counts = {}
    unthemed = 0

    for photo_id, photo in index.get("photos", {}).items():
        themes = photo.get("themes", [])
        if not themes:
            unthemed += 1
        else:
            for theme in themes:
                counts[theme] = counts.get(theme, 0) + 1

    counts["unthemed"] = unthemed
    return counts


def add_predicted_themes(photo_id, themes):
    """
    Add predicted themes to a photo (additive, preserves manual themes).

    Also stores theme_predictions metadata on the photo for auditability.

    Args:
        photo_id: Photo content hash
        themes: List of predicted theme names

    Returns:
        True if successful, False otherwise
    """
    index = config.load_photo_index()

    if photo_id not in index.get("photos", {}):
        return False

    photo = index["photos"][photo_id]

    # Add themes (additive - don't remove existing)
    existing_themes = set(photo.get("themes", []))
    existing_themes.update(themes)
    photo["themes"] = list(existing_themes)

    # Store prediction metadata for auditability
    photo["theme_predictions"] = {
        "predicted_themes": themes,
        "predicted_at": __import__("datetime").datetime.now().isoformat(),
    }

    config.save_photo_index(index)
    return True


def update_photo_quality(photo_id, quality_data):
    """
    Update quality data for a photo.

    Args:
        photo_id: Photo content hash
        quality_data: Dictionary with quality scores from quality_analyzer

    Returns:
        True if successful, False otherwise
    """
    index = config.load_photo_index()

    if photo_id not in index.get("photos", {}):
        return False

    index["photos"][photo_id]["quality"] = quality_data
    config.save_photo_index(index)
    return True


def update_burst_info(burst_groups):
    """
    Update burst group information for photos.

    Args:
        burst_groups: Dictionary mapping burst_group_id to list of photo_ids
    """
    index = config.load_photo_index()

    for burst_id, photo_ids in burst_groups.items():
        # Find best photo in burst
        burst_photos = []
        for pid in photo_ids:
            if pid in index.get("photos", {}):
                burst_photos.append(index["photos"][pid])

        if not burst_photos:
            continue

        # Sort by quality score
        best_photo_id = max(
            burst_photos,
            key=lambda p: p.get("quality", {}).get("overall_score", 0)
        ).get("id")

        # Update all photos in burst
        for pid in photo_ids:
            if pid in index.get("photos", {}):
                if "quality" not in index["photos"][pid]:
                    index["photos"][pid]["quality"] = {}
                index["photos"][pid]["quality"]["burst_group_id"] = burst_id
                index["photos"][pid]["quality"]["is_burst_best"] = (pid == best_photo_id)

    config.save_photo_index(index)


def get_quality_stats(year=None):
    """
    Get quality analysis statistics.

    Args:
        year: Optional year to filter by

    Returns:
        Dictionary with quality statistics
    """
    index = config.load_photo_index()

    total = 0
    analyzed = 0
    good_photos = 0
    burst_groups = set()

    for photo_id, photo in index.get("photos", {}).items():
        if year and str(photo.get("year")) != str(year):
            continue

        total += 1
        quality = photo.get("quality", {})

        if quality.get("analyzed_at"):
            analyzed += 1
            if quality.get("is_good_photo"):
                good_photos += 1
            if quality.get("burst_group_id"):
                burst_groups.add(quality["burst_group_id"])

    return {
        "total_photos": total,
        "analyzed_photos": analyzed,
        "good_photos": good_photos,
        "burst_groups": len(burst_groups),
        "percent_analyzed": round(analyzed / total * 100, 1) if total > 0 else 0,
        "percent_good": round(good_photos / analyzed * 100, 1) if analyzed > 0 else 0,
    }
