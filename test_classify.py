#!/usr/bin/env python3
"""
Test theme classification on local image files.

Usage:
    # Classify a single file
    python test_classify.py ~/Desktop/photo.jpg

    # Classify all images in a directory
    python test_classify.py ~/Desktop/test_photos/

    # Show all class probabilities (not just top)
    python test_classify.py --all-probs ~/Desktop/photo.jpg

    # Use a custom confidence threshold
    python test_classify.py --threshold 0.3 ~/Desktop/test_photos/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from theme_classifier import get_theme_classifier


def classify_file(classifier, img_path, show_all_probs=False):
    """Classify a single image file and print results."""
    with open(img_path, "rb") as f:
        image_bytes = f.read()

    result = classifier.predict(image_bytes)

    theme = result["top_theme"] or "(none)"
    conf = result["top_confidence"]
    assigned = result["predicted_themes"]

    status = f"-> {assigned[0]}" if assigned else "   (below threshold)"
    print(f"  {img_path.name:40s} {theme:15s} {conf:.3f} {status}")

    if show_all_probs:
        # Get full probability breakdown
        features = classifier.extract_features(image_bytes)
        if features is not None:
            import numpy as np
            probas = classifier._classifier.predict_proba(features.reshape(1, -1))[0]
            classes = classifier._label_encoder.classes_
            ranked = sorted(zip(classes, probas), key=lambda x: x[1], reverse=True)
            for cls, prob in ranked:
                bar = "#" * int(prob * 40)
                print(f"      {cls:15s} {prob:.3f} {bar}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Test theme classification on local image files"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Image files or directories to classify"
    )
    parser.add_argument(
        "--all-probs", "-a",
        action="store_true",
        help="Show probabilities for all classes"
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=None,
        help="Override confidence threshold"
    )

    args = parser.parse_args()

    classifier = get_theme_classifier()

    if not classifier.is_available:
        print("Error: Theme classifier not available.")
        print("  Train it first: python train_themes.py <training_dirs>")
        print("  Or install deps: pip install torch torchvision scikit-learn")
        sys.exit(1)

    if args.threshold is not None:
        classifier._confidence_threshold = args.threshold

    meta = classifier.get_metadata()
    print(f"Classifier: {len(meta['classes'])} classes, threshold={classifier._confidence_threshold:.2f}")
    print(f"Classes: {', '.join(meta['classes'])}")
    print()

    # Collect image files
    image_files = []
    for p in args.paths:
        path = Path(p).expanduser()
        if path.is_file():
            image_files.append(path)
        elif path.is_dir():
            for ext in config.IMAGE_EXTENSIONS:
                image_files.extend(path.glob(f"*{ext}"))
                image_files.extend(path.glob(f"*{ext.upper()}"))
        else:
            print(f"Warning: Not found: {p}")

    if not image_files:
        print("No image files found.")
        sys.exit(1)

    image_files.sort()
    print(f"{'File':42s} {'Top Theme':15s} {'Conf':5s} Result")
    print("-" * 80)

    assigned_count = 0
    for img_path in image_files:
        classify_file(classifier, img_path, show_all_probs=args.all_probs)
        with open(img_path, "rb") as f:
            r = classifier.predict(f.read())
        if r["predicted_themes"]:
            assigned_count += 1

    print("-" * 80)
    print(f"Classified: {assigned_count}/{len(image_files)} assigned a theme")


if __name__ == "__main__":
    main()
