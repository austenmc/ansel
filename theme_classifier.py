"""Theme classification using MobileNetV2 features + SVM classifier."""

import io
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

import config

# Optional imports - classifier is gracefully disabled if not available
try:
    import torch
    import torchvision.models as models
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.svm import LinearSVC
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Optional face embedding import
try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False

# Model persistence paths
MODELS_DIR = config.ANSEL_DIR / "models"
MODEL_FILE = MODELS_DIR / "theme_classifier.pkl"
META_FILE = MODELS_DIR / "theme_classifier_meta.json"


class FaceEmbedder:
    """Extract 512-d face embeddings using facenet-pytorch."""

    EMBEDDING_DIM = 512

    def __init__(self):
        self._mtcnn = None
        self._resnet = None

    def _init_models(self):
        """Lazy-load face detection and embedding models."""
        if self._mtcnn is not None:
            return
        self._mtcnn = MTCNN(image_size=160, keep_all=False, device='cpu')
        self._resnet = InceptionResnetV1(pretrained='vggface2').eval()

    def extract(self, image_bytes: bytes) -> np.ndarray:
        """
        Extract face embedding from image.

        Returns 512-d embedding or zeros if no face found.
        """
        if not FACENET_AVAILABLE:
            return np.zeros(self.EMBEDDING_DIM)

        self._init_models()
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            face_tensor = self._mtcnn(img)
            if face_tensor is None:
                return np.zeros(self.EMBEDDING_DIM)
            with torch.no_grad():
                embedding = self._resnet(face_tensor.unsqueeze(0))
            return embedding.squeeze().numpy()
        except Exception:
            return np.zeros(self.EMBEDDING_DIM)


def encode_temporal_features(date_taken: Optional[datetime]) -> np.ndarray:
    """
    Encode month as cyclical sin/cos (2-d).

    Returns zeros if no date provided.
    """
    if date_taken is None:
        return np.array([0.0, 0.0])
    angle = 2 * np.pi * (date_taken.month - 1) / 12
    return np.array([np.sin(angle), np.cos(angle)])


class ThemeClassifier:
    """Image theme classifier using MobileNetV2 features + SVM."""

    DEFAULT_CONFIDENCE_THRESHOLD = 0.45

    def __init__(self):
        self._feature_extractor = None
        self._transform = None
        self._classifier = None
        self._label_encoder = None
        self._confidence_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD
        self._metadata = None
        self._model_loaded = False
        self._face_embedder = FaceEmbedder() if FACENET_AVAILABLE else None

    @property
    def is_available(self) -> bool:
        """Check if classifier is usable (deps installed + model trained)."""
        if not TORCH_AVAILABLE or not SKLEARN_AVAILABLE:
            return False
        if not self._model_loaded:
            self._load_model()
        return self._classifier is not None

    def _init_feature_extractor(self):
        """Lazy-load MobileNetV2 as feature extractor."""
        if self._feature_extractor is not None:
            return

        if not TORCH_AVAILABLE:
            return

        # Load pretrained MobileNetV2, remove classifier head
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        model.classifier = torch.nn.Identity()
        model.eval()
        self._feature_extractor = model

        # Standard ImageNet transforms
        self._transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract_features(self, image_bytes: bytes, date_taken: Optional[datetime] = None) -> Optional[np.ndarray]:
        """
        Extract combined feature vector from image bytes.

        Features include:
        - MobileNetV2 visual features (1280-d)
        - Face embedding (512-d, zeros if no face)
        - Temporal features (2-d, cyclical month encoding)

        Args:
            image_bytes: Raw image file bytes
            date_taken: Optional datetime for temporal encoding

        Returns:
            1794-dimensional numpy array, or None on failure
        """
        if not TORCH_AVAILABLE:
            return None

        self._init_feature_extractor()

        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")

            input_tensor = self._transform(img).unsqueeze(0)

            with torch.no_grad():
                visual_features = self._feature_extractor(input_tensor)

            visual_features = visual_features.squeeze().numpy()

            # Face embedding (512-d)
            if self._face_embedder:
                face_features = self._face_embedder.extract(image_bytes)
            else:
                face_features = np.zeros(FaceEmbedder.EMBEDDING_DIM)

            # Temporal features (2-d)
            temporal_features = encode_temporal_features(date_taken)

            # Concatenate: 1280 + 512 + 2 = 1794
            return np.concatenate([visual_features, face_features, temporal_features])
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None

    def predict(self, image_bytes: bytes, date_taken: Optional[datetime] = None) -> dict:
        """
        Predict theme for an image.

        Args:
            image_bytes: Raw image file bytes
            date_taken: Optional datetime for temporal features

        Returns:
            Dictionary with predicted_themes, top_theme, top_confidence
        """
        empty_result = {"predicted_themes": [], "top_theme": None, "top_confidence": 0.0}

        if not self.is_available:
            return empty_result

        features = self.extract_features(image_bytes, date_taken=date_taken)
        if features is None:
            return empty_result

        try:
            # Get probability predictions
            probas = self._classifier.predict_proba(features.reshape(1, -1))[0]
            classes = self._label_encoder.classes_

            # Find top prediction
            top_idx = np.argmax(probas)
            top_theme = classes[top_idx]
            top_confidence = float(probas[top_idx])

            return {
                "predicted_themes": [top_theme],
                "top_theme": top_theme,
                "top_confidence": round(top_confidence, 4),
            }
        except Exception as e:
            print(f"Error predicting theme: {e}")
            return empty_result

    def train(self, training_dirs: list, min_samples: int = 10, verbose: bool = False) -> dict:
        """
        Train the classifier from organized photo folders.

        Args:
            training_dirs: List of paths to training directories (each containing theme subfolders)
            min_samples: Minimum samples per class to include
            verbose: Print detailed output

        Returns:
            Training statistics dictionary
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("torch/torchvision not installed")
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn not installed")

        from sklearn.model_selection import cross_val_score, StratifiedKFold

        # Collect training data
        samples = []  # (image_path, label)
        for train_dir in training_dirs:
            train_path = Path(train_dir)
            if not train_path.exists():
                print(f"Warning: Directory not found: {train_dir}")
                continue

            for theme_folder in sorted(train_path.iterdir()):
                if not theme_folder.is_dir():
                    continue

                label = self._normalize_label(theme_folder.name)
                if label is None:
                    if verbose:
                        print(f"  Excluding: {theme_folder.name}")
                    continue

                # Find images in this folder
                image_files = []
                for ext in config.IMAGE_EXTENSIONS:
                    image_files.extend(theme_folder.glob(f"*{ext}"))
                    image_files.extend(theme_folder.glob(f"*{ext.upper()}"))

                for img_path in image_files:
                    samples.append((img_path, label))

        if not samples:
            raise ValueError("No training samples found")

        # Count per class and filter by min_samples
        class_counts = {}
        for _, label in samples:
            class_counts[label] = class_counts.get(label, 0) + 1

        if verbose:
            print(f"\nClass distribution (before filtering):")
            for label, count in sorted(class_counts.items()):
                print(f"  {label}: {count}")

        valid_classes = {label for label, count in class_counts.items() if count >= min_samples}
        filtered_samples = [(path, label) for path, label in samples if label in valid_classes]

        excluded = {label: count for label, count in class_counts.items() if count < min_samples}
        if excluded and verbose:
            print(f"\nExcluded (< {min_samples} samples):")
            for label, count in sorted(excluded.items()):
                print(f"  {label}: {count}")

        if not filtered_samples:
            raise ValueError(f"No classes with >= {min_samples} samples")

        # Extract features
        self._init_feature_extractor()
        features_list = []
        labels_list = []
        errors = 0

        # Import for EXIF extraction
        from photo_service import extract_exif_date

        print(f"\nExtracting features from {len(filtered_samples)} images...")

        for i, (img_path, label) in enumerate(filtered_samples):
            try:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()

                # Extract EXIF date for temporal features
                date_taken = extract_exif_date(image_bytes)

                feat = self.extract_features(image_bytes, date_taken=date_taken)
                if feat is not None:
                    features_list.append(feat)
                    labels_list.append(label)
                else:
                    errors += 1
            except Exception as e:
                errors += 1
                if verbose:
                    print(f"  Error: {img_path.name}: {e}")

            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{len(filtered_samples)}...")

        if len(features_list) < 2:
            raise ValueError("Not enough features extracted for training")

        X = np.array(features_list)
        y = np.array(labels_list)

        # Encode labels
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)

        # Train LinearSVC + CalibratedClassifierCV
        print(f"\nTraining classifier on {len(X)} samples, {len(self._label_encoder.classes_)} classes...")
        base_svc = LinearSVC(class_weight="balanced", max_iter=5000)
        self._classifier = CalibratedClassifierCV(base_svc, cv=3)
        self._classifier.fit(X, y_encoded)

        # Cross-validation
        print("Running cross-validation...")
        skf = StratifiedKFold(n_splits=min(5, min(np.bincount(y_encoded))), shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000), cv=3),
            X, y_encoded, cv=skf, scoring="accuracy"
        )

        # Auto-calibrate confidence threshold
        self._confidence_threshold = self._calibrate_threshold(X, y_encoded, target_precision=0.8)

        # Build metadata
        final_counts = {}
        for label in labels_list:
            final_counts[label] = final_counts.get(label, 0) + 1

        self._metadata = {
            "trained_at": datetime.now().isoformat(),
            "classes": list(self._label_encoder.classes_),
            "class_counts": final_counts,
            "total_samples": len(X),
            "cv_accuracy": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "confidence_threshold": self._confidence_threshold,
            "errors": errors,
            "training_dirs": [str(d) for d in training_dirs],
            "feature_dimensions": {
                "mobilenet": 1280,
                "face": FaceEmbedder.EMBEDDING_DIM if FACENET_AVAILABLE else 0,
                "temporal": 2,
            },
            "total_features": X.shape[1],
        }

        # Save model
        self._save_model()
        self._model_loaded = True

        # Print results
        print(f"\nTraining complete:")
        print(f"  Classes: {len(self._label_encoder.classes_)}")
        print(f"  Samples: {len(X)}")
        print(f"  CV Accuracy: {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
        print(f"  Confidence threshold: {self._confidence_threshold:.3f}")
        if errors:
            print(f"  Errors: {errors}")

        return self._metadata

    def _normalize_label(self, folder_name: str) -> Optional[str]:
        """
        Normalize a folder name to a training label.

        Returns None if the folder should be excluded.
        """
        # Exclusions
        excluded = {"Random", "Remodel", "San Diego", "Yosemite", "Norment Vacation"}
        if folder_name in excluded:
            return None

        # Birthday merges: "X's BDay" or "Xs BDay" -> "X"
        name = folder_name
        for suffix in ["'s BDay", "s BDay", "'s Bday", "s Bday", "'s Birthday", "s Birthday"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
                break

        return name

    def _calibrate_threshold(self, X: np.ndarray, y: np.ndarray, target_precision: float = 0.8) -> float:
        """
        Auto-calibrate confidence threshold to achieve target precision.

        Uses leave-one-out style prediction on training data.
        """
        from sklearn.model_selection import cross_val_predict

        try:
            # Get cross-validated probability predictions
            base_svc = LinearSVC(class_weight="balanced", max_iter=5000)
            cal_svc = CalibratedClassifierCV(base_svc, cv=3)

            skf = StratifiedKFold(n_splits=min(5, min(np.bincount(y))), shuffle=True, random_state=42)
            probas = cross_val_predict(cal_svc, X, y, cv=skf, method="predict_proba")

            # Try different thresholds
            best_threshold = self.DEFAULT_CONFIDENCE_THRESHOLD
            for threshold in np.arange(0.3, 0.8, 0.05):
                top_probas = probas.max(axis=1)
                top_preds = probas.argmax(axis=1)

                # Filter predictions above threshold
                mask = top_probas >= threshold
                if mask.sum() == 0:
                    continue

                precision = (top_preds[mask] == y[mask]).mean()
                coverage = mask.mean()

                if precision >= target_precision and coverage >= 0.3:
                    best_threshold = float(threshold)
                    break

            return round(best_threshold, 2)
        except Exception:
            return self.DEFAULT_CONFIDENCE_THRESHOLD

    def _load_model(self):
        """Load trained model from disk."""
        if self._model_loaded:
            return

        if not SKLEARN_AVAILABLE:
            return

        if not MODEL_FILE.exists():
            return

        try:
            with open(MODEL_FILE, "rb") as f:
                data = pickle.load(f)

            self._classifier = data["classifier"]
            self._label_encoder = data["label_encoder"]
            self._confidence_threshold = data.get("confidence_threshold", self.DEFAULT_CONFIDENCE_THRESHOLD)
            self._model_loaded = True

            # Load metadata
            if META_FILE.exists():
                with open(META_FILE) as f:
                    self._metadata = json.load(f)

        except Exception as e:
            print(f"Error loading theme classifier model: {e}")
            self._classifier = None
            self._label_encoder = None

    def _save_model(self):
        """Save trained model to disk."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Save classifier + label encoder + threshold
        model_data = {
            "classifier": self._classifier,
            "label_encoder": self._label_encoder,
            "confidence_threshold": self._confidence_threshold,
        }
        with open(MODEL_FILE, "wb") as f:
            pickle.dump(model_data, f)

        # Save metadata as JSON
        if self._metadata:
            with open(META_FILE, "w") as f:
                json.dump(self._metadata, f, indent=2)

    def get_metadata(self) -> Optional[dict]:
        """Get training metadata."""
        if not self._model_loaded:
            self._load_model()
        return self._metadata


# Singleton instance
_theme_classifier = None


def get_theme_classifier() -> ThemeClassifier:
    """Get or create singleton ThemeClassifier instance."""
    global _theme_classifier
    if _theme_classifier is None:
        _theme_classifier = ThemeClassifier()
    return _theme_classifier


def classify_photo(image_bytes: bytes, photo_id: str, date_taken: Optional[datetime] = None) -> dict:
    """
    Classify a photo's theme.

    Convenience function using the singleton classifier.

    Args:
        image_bytes: Raw image file bytes
        photo_id: Photo ID for reference
        date_taken: Optional datetime for temporal features

    Returns:
        Dictionary with predicted_themes, top_theme, top_confidence
    """
    classifier = get_theme_classifier()
    return classifier.predict(image_bytes, date_taken=date_taken)
