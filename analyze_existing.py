#!/usr/bin/env python3
"""
Standalone script to analyze photo quality for existing cached thumbnails.

Usage:
    # Analyze all years
    python analyze_existing.py

    # Analyze specific year
    python analyze_existing.py --year 2023

    # Re-analyze even if already scored
    python analyze_existing.py --force

    # Analyze with verbose output
    python analyze_existing.py --verbose
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import config
import photo_service
import quality_analyzer
from quality_analyzer import BurstDetector
from theme_classifier import get_theme_classifier

# Try to import tqdm for progress bar, fall back to simple output
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("Note: Install tqdm for progress bars (pip install tqdm)")


def analyze_year(year, force=False, verbose=False, classify_themes=False):
    """
    Analyze all photos for a specific year.

    Args:
        year: Year to analyze
        force: If True, re-analyze even if already scored
        verbose: If True, print detailed output
        classify_themes: If True, also run theme classification

    Returns:
        Dictionary with analysis statistics
    """
    photos = photo_service.get_photos_for_year(year)

    # Filter to photos with thumbnails
    photos_with_thumbnails = [p for p in photos if p.get("has_thumbnail")]

    if not photos_with_thumbnails:
        print(f"  No thumbnails found for {year}")
        return {"analyzed": 0, "skipped": 0, "errors": 0}

    # Filter to photos needing analysis (unless force)
    if force:
        photos_to_analyze = photos_with_thumbnails
    else:
        photos_to_analyze = [
            p for p in photos_with_thumbnails
            if not p.get("quality", {}).get("analyzed_at")
        ]

    if not photos_to_analyze:
        print(f"  All {len(photos_with_thumbnails)} photos already analyzed for {year}")
        return {"analyzed": 0, "skipped": len(photos_with_thumbnails), "errors": 0}

    print(f"  Analyzing {len(photos_to_analyze)} photos for {year}...")

    scorer = quality_analyzer.get_quality_scorer()
    analyzed = 0
    errors = 0

    # Use tqdm if available
    if TQDM_AVAILABLE:
        iterator = tqdm(photos_to_analyze, desc=f"  {year}", unit="photo")
    else:
        iterator = photos_to_analyze

    for i, photo in enumerate(iterator):
        try:
            # Read thumbnail from cache
            thumbnail_path = config.THUMBNAILS_DIR / f"{photo['id']}.jpg"
            if not thumbnail_path.exists():
                continue

            with open(thumbnail_path, "rb") as f:
                image_bytes = f.read()

            # Analyze quality
            quality_data = scorer.analyze_photo(image_bytes, photo["id"])

            # Save to index
            photo_service.update_photo_quality(photo["id"], quality_data)

            # Classify theme if requested
            if classify_themes:
                classifier = get_theme_classifier()
                if classifier.is_available:
                    # Lookup date_taken from photo data for temporal features
                    date_str = photo.get("date_taken")
                    date_taken = datetime.fromisoformat(date_str) if date_str else None

                    theme_result = classifier.predict(image_bytes, date_taken=date_taken)
                    if theme_result.get("predicted_themes"):
                        photo_service.add_predicted_themes(photo["id"], theme_result["predicted_themes"])

            analyzed += 1

            if verbose and not TQDM_AVAILABLE:
                score = quality_data.get("overall_score", 0)
                good = "GOOD" if quality_data.get("is_good_photo") else ""
                print(f"    {photo['name']}: {score:.1f} {good}")

            # Progress update for non-tqdm
            if not TQDM_AVAILABLE and (i + 1) % 50 == 0:
                print(f"    Analyzed {i + 1}/{len(photos_to_analyze)}...")

        except Exception as e:
            errors += 1
            if verbose:
                print(f"    Error analyzing {photo.get('name', photo['id'])}: {e}")

    skipped = len(photos_with_thumbnails) - len(photos_to_analyze)
    return {"analyzed": analyzed, "skipped": skipped, "errors": errors}


def detect_bursts_for_year(year, verbose=False):
    """
    Detect and mark burst groups for a year.

    Args:
        year: Year to process
        verbose: If True, print detailed output

    Returns:
        Number of burst groups found
    """
    photos = photo_service.get_photos_for_year(year)

    if not photos:
        return 0

    detector = BurstDetector()
    burst_groups = detector.detect_bursts(photos)

    if burst_groups:
        photo_service.update_burst_info(burst_groups)
        if verbose:
            for burst_id, photo_ids in burst_groups.items():
                print(f"    {burst_id}: {len(photo_ids)} photos")

    return len(burst_groups)


def print_quality_distribution(year=None):
    """Print distribution of quality scores."""
    index = config.load_photo_index()

    scores = []
    for photo_id, photo in index.get("photos", {}).items():
        if year and str(photo.get("year")) != str(year):
            continue

        quality = photo.get("quality", {})
        if quality.get("overall_score"):
            scores.append(quality["overall_score"])

    if not scores:
        print("  No quality scores available")
        return

    scores.sort()

    # Calculate percentiles
    def percentile(data, p):
        idx = int(len(data) * p / 100)
        return data[min(idx, len(data) - 1)]

    print(f"\n  Quality Score Distribution ({len(scores)} photos):")
    print(f"    Min:  {min(scores):.1f}")
    print(f"    25th: {percentile(scores, 25):.1f}")
    print(f"    50th: {percentile(scores, 50):.1f}")
    print(f"    75th: {percentile(scores, 75):.1f}")
    print(f"    Max:  {max(scores):.1f}")

    # Count good vs not good
    stats = photo_service.get_quality_stats(year)
    print(f"\n  Good photos: {stats['good_photos']} ({stats['percent_good']:.1f}%)")
    print(f"  Burst groups: {stats['burst_groups']}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze photo quality for existing cached thumbnails"
    )
    parser.add_argument(
        "--year", "-y",
        type=int,
        help="Analyze only this year"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Re-analyze even if already scored"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output"
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print statistics, don't analyze"
    )
    parser.add_argument(
        "--classify-themes",
        action="store_true",
        help="Also run theme classification on thumbnails"
    )

    args = parser.parse_args()

    print("Ansel Photo Quality Analyzer")
    print("=" * 40)

    # Get years to process
    if args.year:
        years = [args.year]
    else:
        years_data = photo_service.get_years()
        years = [y["year"] for y in years_data]

    if not years:
        print("No years found in photo index.")
        print("Run the app and scan your Dropbox first.")
        return

    if args.stats_only:
        for year in sorted(years, reverse=True):
            print(f"\nYear {year}:")
            print_quality_distribution(year)
        return

    # Check theme classifier availability
    if args.classify_themes:
        classifier = get_theme_classifier()
        if not classifier.is_available:
            print("Warning: Theme classifier not available.")
            print("  Train it first: python train_themes.py <training_dirs>")
            print("  Or install deps: pip install torch torchvision scikit-learn")
            print()

    # Analyze each year
    total_analyzed = 0
    total_skipped = 0
    total_errors = 0
    total_bursts = 0

    for year in sorted(years, reverse=True):
        print(f"\nYear {year}:")

        # Analyze quality
        stats = analyze_year(year, force=args.force, verbose=args.verbose, classify_themes=args.classify_themes)
        total_analyzed += stats["analyzed"]
        total_skipped += stats["skipped"]
        total_errors += stats["errors"]

        # Detect bursts
        if stats["analyzed"] > 0 or args.force:
            bursts = detect_bursts_for_year(year, verbose=args.verbose)
            total_bursts += bursts
            if bursts > 0:
                print(f"  Found {bursts} burst groups")

        # Print distribution for this year
        if args.verbose:
            print_quality_distribution(year)

    # Summary
    print("\n" + "=" * 40)
    print("Summary:")
    print(f"  Analyzed: {total_analyzed}")
    print(f"  Skipped (already analyzed): {total_skipped}")
    print(f"  Errors: {total_errors}")
    print(f"  Burst groups: {total_bursts}")

    if total_analyzed > 0:
        print("\nQuality data saved to ~/.ansel/photo_index.json")


if __name__ == "__main__":
    main()
