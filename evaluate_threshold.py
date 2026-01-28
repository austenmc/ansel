#!/usr/bin/env python3
"""Quick evaluation at different thresholds - efficient version."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config
from photo_service import extract_exif_date
from theme_classifier import get_theme_classifier

def normalize_label(name):
    for suffix in ["'s BDay", "s BDay"]:
        if name.endswith(suffix):
            return name[:-len(suffix)]
    return name

test_dir = sys.argv[1] if len(sys.argv) > 1 else "/Users/austenmc/Desktop/2020 Photos/Done/"

classifier = get_theme_classifier()
metadata = classifier.get_metadata()
classes = set(metadata.get('classes', []))

print(f"Processing {test_dir}...")
print(f"Classes: {classes}")
print()

# Single pass - collect all results
results = []  # (true_label, top_pred, confidence)
test_path = Path(test_dir)

for theme_folder in sorted(test_path.iterdir()):
    if not theme_folder.is_dir():
        continue

    true_label = normalize_label(theme_folder.name)
    if true_label not in classes:
        print(f"Skipping {theme_folder.name} (not in training)")
        continue

    image_files = []
    for ext in config.IMAGE_EXTENSIONS:
        image_files.extend(theme_folder.glob(f"*{ext}"))
        image_files.extend(theme_folder.glob(f"*{ext.upper()}"))

    print(f"Processing {theme_folder.name} ({len(image_files)} images)...", flush=True)

    for img_path in image_files:
        try:
            with open(img_path, "rb") as f:
                image_bytes = f.read()
            date_taken = extract_exif_date(image_bytes)
            result = classifier.predict(image_bytes, date_taken=date_taken)
            results.append((true_label, result["top_theme"], result["top_confidence"]))
        except Exception as e:
            pass

print(f"\nProcessed {len(results)} images total")
print()

# Evaluate at different thresholds
print(f"{'Threshold':>10} {'Coverage':>10} {'Precision':>10} {'Recall':>10} {'Predicted':>10} {'Correct':>10}")
print("-" * 65)

for thresh in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:
    total = len(results)
    predicted = [(t, p, c) for t, p, c in results if c >= thresh]
    correct = sum(1 for t, p, c in predicted if t == p)

    coverage = len(predicted) / total * 100 if total else 0
    precision = correct / len(predicted) * 100 if predicted else 0
    recall = correct / total * 100 if total else 0

    print(f"{thresh:>10.2f} {coverage:>9.1f}% {precision:>9.1f}% {recall:>9.1f}% {len(predicted):>10} {correct:>10}")
