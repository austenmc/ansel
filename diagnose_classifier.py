#!/usr/bin/env python3
"""Diagnose classifier performance issues."""

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import config
from photo_service import extract_exif_date
from theme_classifier import get_theme_classifier, FACENET_AVAILABLE

def diagnose(test_dir: str):
    classifier = get_theme_classifier()
    metadata = classifier.get_metadata()

    print("=== CLASSIFIER INFO ===")
    print(f"Confidence threshold: {classifier._confidence_threshold}")
    print(f"Classes: {metadata.get('classes')}")
    print(f"Training samples: {metadata.get('total_samples')}")
    print(f"Feature dims: {metadata.get('feature_dimensions')}")
    print(f"FaceNet available: {FACENET_AVAILABLE}")
    print()

    test_path = Path(test_dir)

    # Collect confidence scores and face detection stats
    confidences = []
    face_detected = 0
    total_images = 0
    per_class_conf = defaultdict(list)

    print("=== ANALYZING TEST SET ===")

    for theme_folder in sorted(test_path.iterdir()):
        if not theme_folder.is_dir():
            continue

        true_label = theme_folder.name
        # Normalize
        for suffix in ["'s BDay", "s BDay"]:
            if true_label.endswith(suffix):
                true_label = true_label[:-len(suffix)]
                break

        image_files = []
        for ext in config.IMAGE_EXTENSIONS:
            image_files.extend(theme_folder.glob(f"*{ext}"))
            image_files.extend(theme_folder.glob(f"*{ext.upper()}"))

        for img_path in list(image_files)[:10]:  # Sample 10 per class
            try:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()

                date_taken = extract_exif_date(image_bytes)

                # Get raw features to check face detection
                features = classifier.extract_features(image_bytes, date_taken=date_taken)
                if features is not None:
                    # Face features are indices 1280:1792
                    face_features = features[1280:1792]
                    has_face = np.any(face_features != 0)
                    if has_face:
                        face_detected += 1

                result = classifier.predict(image_bytes, date_taken=date_taken)
                conf = result.get("top_confidence", 0)
                confidences.append(conf)
                per_class_conf[true_label].append(conf)
                total_images += 1

            except Exception as e:
                print(f"  Error: {img_path.name}: {e}")

    print(f"\nAnalyzed {total_images} images")
    print(f"Face detected: {face_detected}/{total_images} ({face_detected/total_images*100:.1f}%)")

    print("\n=== CONFIDENCE DISTRIBUTION ===")
    confidences = np.array(confidences)
    print(f"Min: {confidences.min():.3f}")
    print(f"25th percentile: {np.percentile(confidences, 25):.3f}")
    print(f"Median: {np.median(confidences):.3f}")
    print(f"75th percentile: {np.percentile(confidences, 75):.3f}")
    print(f"Max: {confidences.max():.3f}")

    print(f"\n% above threshold ({classifier._confidence_threshold}):",
          f"{(confidences >= classifier._confidence_threshold).mean()*100:.1f}%")

    print("\nAt different thresholds:")
    for thresh in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        pct = (confidences >= thresh).mean() * 100
        print(f"  >= {thresh}: {pct:.1f}% would get predictions")

    print("\n=== PER-CLASS CONFIDENCE ===")
    for label in sorted(per_class_conf.keys()):
        confs = per_class_conf[label]
        if confs:
            print(f"  {label}: median={np.median(confs):.3f}, max={np.max(confs):.3f}")

if __name__ == "__main__":
    diagnose(sys.argv[1] if len(sys.argv) > 1 else "/Users/austenmc/Desktop/2020 Photos/Done/")
