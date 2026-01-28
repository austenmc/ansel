#!/usr/bin/env python3
"""
Evaluate theme classifier on a held-out test set.

Usage:
    python evaluate_classifier.py /path/to/test/folder
    python evaluate_classifier.py --verbose /path/to/test/folder
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from photo_service import extract_exif_date
from theme_classifier import get_theme_classifier, ThemeClassifier


def normalize_label(folder_name: str) -> str:
    """Normalize folder name to match training labels."""
    # Use same normalization as training
    name = folder_name
    for suffix in ["'s BDay", "s BDay", "'s Bday", "s Bday", "'s Birthday", "s Birthday"]:
        if name.endswith(suffix):
            name = name[:-len(suffix)]
            break
    return name


def evaluate(test_dir: str, verbose: bool = False):
    """Evaluate classifier on test directory."""
    classifier = get_theme_classifier()

    if not classifier.is_available:
        print("Error: Classifier not available. Train it first.")
        sys.exit(1)

    metadata = classifier.get_metadata()
    print(f"Classifier trained on: {metadata.get('total_samples')} samples")
    print(f"Classes: {', '.join(metadata.get('classes', []))}")
    print(f"Feature dimensions: {metadata.get('total_features', 'unknown')}")
    print()

    test_path = Path(test_dir)
    if not test_path.exists():
        print(f"Error: Directory not found: {test_dir}")
        sys.exit(1)

    # Collect results
    results = []  # (true_label, predicted_label, confidence, path)
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_predicted = defaultdict(int)

    # Find all theme folders
    theme_folders = [f for f in sorted(test_path.iterdir()) if f.is_dir()]

    print(f"Evaluating {len(theme_folders)} theme folders...")
    print()

    for theme_folder in theme_folders:
        true_label = normalize_label(theme_folder.name)

        # Skip if this class wasn't in training
        if true_label not in metadata.get('classes', []):
            print(f"  Skipping {theme_folder.name} (not in training set)")
            continue

        # Find images
        image_files = []
        for ext in config.IMAGE_EXTENSIONS:
            image_files.extend(theme_folder.glob(f"*{ext}"))
            image_files.extend(theme_folder.glob(f"*{ext.upper()}"))

        if not image_files:
            continue

        folder_correct = 0
        folder_total = 0

        for img_path in image_files:
            try:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()

                # Extract date for temporal features
                date_taken = extract_exif_date(image_bytes)

                # Predict
                result = classifier.predict(image_bytes, date_taken=date_taken)
                predicted = result.get("top_theme")
                confidence = result.get("top_confidence", 0)

                # Only count if classifier made a prediction
                assigned = result.get("predicted_themes", [])

                class_total[true_label] += 1
                folder_total += 1

                if assigned:
                    class_predicted[predicted] += 1
                    if predicted == true_label:
                        class_correct[true_label] += 1
                        folder_correct += 1
                        if verbose:
                            print(f"    ✓ {img_path.name}: {predicted} ({confidence:.2f})")
                    else:
                        if verbose:
                            print(f"    ✗ {img_path.name}: predicted {predicted} ({confidence:.2f}), actual {true_label}")
                else:
                    if verbose:
                        print(f"    - {img_path.name}: no prediction (top: {predicted} @ {confidence:.2f})")

                results.append((true_label, predicted if assigned else None, confidence, str(img_path)))

            except Exception as e:
                if verbose:
                    print(f"    Error: {img_path.name}: {e}")

        if folder_total > 0:
            acc = folder_correct / folder_total * 100
            print(f"  {theme_folder.name}: {folder_correct}/{folder_total} ({acc:.1f}%)")

    # Overall metrics
    print()
    print("=" * 50)
    print("OVERALL RESULTS")
    print("=" * 50)

    total_samples = sum(class_total.values())
    total_correct = sum(class_correct.values())
    total_predicted = len([r for r in results if r[1] is not None])

    print(f"\nTotal test samples: {total_samples}")
    print(f"Predictions made: {total_predicted} ({total_predicted/total_samples*100:.1f}% coverage)")
    print(f"Correct predictions: {total_correct}")

    if total_predicted > 0:
        precision = total_correct / total_predicted * 100
        print(f"Precision (when predicting): {precision:.1f}%")

    if total_samples > 0:
        recall = total_correct / total_samples * 100
        print(f"Recall (overall accuracy): {recall:.1f}%")

    # Per-class breakdown
    print("\nPer-class performance:")
    print(f"{'Class':<15} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 45)

    for label in sorted(class_total.keys()):
        correct = class_correct[label]
        total = class_total[label]
        acc = correct / total * 100 if total > 0 else 0
        print(f"{label:<15} {correct:>8} {total:>8} {acc:>9.1f}%")

    # Confusion matrix (abbreviated - just show misclassifications)
    print("\nCommon misclassifications:")
    confusion = defaultdict(lambda: defaultdict(int))
    for true_label, pred_label, _, _ in results:
        if pred_label and pred_label != true_label:
            confusion[true_label][pred_label] += 1

    for true_label in sorted(confusion.keys()):
        misses = sorted(confusion[true_label].items(), key=lambda x: -x[1])[:3]
        if misses:
            miss_str = ", ".join(f"{p}({c})" for p, c in misses)
            print(f"  {true_label} -> {miss_str}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate theme classifier on test set")
    parser.add_argument("test_dir", help="Path to test directory with theme subfolders")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-image results")

    args = parser.parse_args()
    evaluate(args.test_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
