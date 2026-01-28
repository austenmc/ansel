#!/usr/bin/env python3
"""
Train the theme classifier from organized photo folders.

Usage:
    python train_themes.py ~/Desktop/"2021 Photos" ~/Desktop/"2022 Photos"
    python train_themes.py --min-samples 10 --verbose ~/Desktop/"2021 Photos"
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def main():
    parser = argparse.ArgumentParser(
        description="Train the Ansel theme classifier from organized photo folders"
    )
    parser.add_argument(
        "training_dirs",
        nargs="+",
        help="Paths to training directories (each containing theme subfolders)"
    )
    parser.add_argument(
        "--min-samples", "-m",
        type=int,
        default=10,
        help="Minimum samples per class to include (default: 10)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Print detailed output"
    )

    args = parser.parse_args()

    # Validate dependencies
    try:
        import torch
        import torchvision
    except ImportError:
        print("Error: torch and torchvision are required for training.")
        print("Install with: pip install torch torchvision")
        sys.exit(1)

    try:
        import sklearn
    except ImportError:
        print("Error: scikit-learn is required for training.")
        print("Install with: pip install scikit-learn")
        sys.exit(1)

    # Validate training directories
    valid_dirs = []
    for d in args.training_dirs:
        p = Path(d).expanduser()
        if p.exists():
            valid_dirs.append(str(p))
        else:
            print(f"Warning: Directory not found: {d}")

    if not valid_dirs:
        print("Error: No valid training directories found.")
        sys.exit(1)

    print("Ansel Theme Classifier Training")
    print("=" * 40)
    print(f"Training directories: {valid_dirs}")
    print(f"Min samples per class: {args.min_samples}")
    print()

    # Train
    from theme_classifier import get_theme_classifier

    classifier = get_theme_classifier()
    try:
        stats = classifier.train(
            training_dirs=valid_dirs,
            min_samples=args.min_samples,
            verbose=args.verbose,
        )

        print(f"\nModel saved to: ~/.ansel/models/")
        print(f"Classes: {', '.join(stats['classes'])}")

    except Exception as e:
        print(f"\nTraining failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
