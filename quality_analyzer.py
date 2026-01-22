"""AI-based photo quality analysis using MediaPipe and OpenCV."""

import io
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

# MediaPipe is optional - face analysis will be skipped if not available
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False

import config


class FaceAnalyzer:
    """Analyze face quality using MediaPipe Face Mesh."""

    # Eye landmark indices for EAR calculation
    LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]

    # Mouth landmark indices for smile detection
    MOUTH_OUTER_INDICES = [61, 291, 0, 17]  # corners and top/bottom

    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            self.face_mesh = None
            return

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=10,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, img_rgb: np.ndarray) -> dict:
        """
        Analyze faces in an image.

        Args:
            img_rgb: RGB image as numpy array

        Returns:
            Dictionary with face analysis results
        """
        if not MEDIAPIPE_AVAILABLE or self.face_mesh is None:
            return {
                "has_faces": False,
                "face_count": 0,
                "face_quality_score": 0,
                "best_eyes_open": 0,
                "best_expression": 0,
            }

        results = self.face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return {
                "has_faces": False,
                "face_count": 0,
                "face_quality_score": 0,
                "best_eyes_open": 0,
                "best_expression": 0,
            }

        face_count = len(results.multi_face_landmarks)
        best_ear = 0
        best_smile = 0
        face_scores = []

        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            # Calculate Eye Aspect Ratio (EAR)
            left_ear = self._calculate_ear(landmarks, self.LEFT_EYE_INDICES)
            right_ear = self._calculate_ear(landmarks, self.RIGHT_EYE_INDICES)
            avg_ear = (left_ear + right_ear) / 2

            # Calculate smile score
            smile_score = self._calculate_smile(landmarks)

            # Track best values
            best_ear = max(best_ear, avg_ear)
            best_smile = max(best_smile, smile_score)

            # Calculate face quality score
            # EAR > 0.25 is open eyes, < 0.15 is closed
            eyes_score = min(100, (avg_ear / 0.3) * 100)
            expression_score = smile_score * 100

            face_quality = eyes_score * 0.6 + expression_score * 0.4
            face_scores.append(face_quality)

        # Overall face quality is average of all faces
        avg_face_quality = sum(face_scores) / len(face_scores) if face_scores else 0

        return {
            "has_faces": True,
            "face_count": int(face_count),
            "face_quality_score": float(round(avg_face_quality, 1)),
            "best_eyes_open": float(round(best_ear, 3)),
            "best_expression": float(round(best_smile, 3)),
        }

    def _calculate_ear(self, landmarks, eye_indices) -> float:
        """Calculate Eye Aspect Ratio."""
        try:
            # Get eye landmarks
            p1 = landmarks[eye_indices[0]]
            p2 = landmarks[eye_indices[1]]
            p3 = landmarks[eye_indices[2]]
            p4 = landmarks[eye_indices[3]]
            p5 = landmarks[eye_indices[4]]
            p6 = landmarks[eye_indices[5]]

            # Vertical distances
            v1 = ((p2.x - p6.x) ** 2 + (p2.y - p6.y) ** 2) ** 0.5
            v2 = ((p3.x - p5.x) ** 2 + (p3.y - p5.y) ** 2) ** 0.5

            # Horizontal distance
            h = ((p1.x - p4.x) ** 2 + (p1.y - p4.y) ** 2) ** 0.5

            if h == 0:
                return 0

            ear = (v1 + v2) / (2.0 * h)
            return ear
        except (IndexError, AttributeError):
            return 0

    def _calculate_smile(self, landmarks) -> float:
        """Calculate smile score based on mouth shape."""
        try:
            # Mouth corners
            left_corner = landmarks[61]
            right_corner = landmarks[291]

            # Mouth top and bottom
            top = landmarks[0]
            bottom = landmarks[17]

            # Width of mouth
            width = ((right_corner.x - left_corner.x) ** 2 +
                     (right_corner.y - left_corner.y) ** 2) ** 0.5

            # Height of mouth
            height = ((top.x - bottom.x) ** 2 +
                      (top.y - bottom.y) ** 2) ** 0.5

            if height == 0:
                return 0

            # Higher width-to-height ratio suggests a smile
            ratio = width / height
            # Normalize: ratio > 3 is a big smile
            smile_score = min(1.0, ratio / 3.0)
            return smile_score
        except (IndexError, AttributeError):
            return 0

    def close(self):
        """Release resources."""
        if self.face_mesh:
            self.face_mesh.close()


class QualityScorer:
    """Score photo quality based on multiple factors."""

    # Default weights for photos WITH faces
    WEIGHTS_WITH_FACES = {
        "sharpness": 0.30,
        "face_quality": 0.25,
        "exposure": 0.20,
        "composition": 0.15,
        "color": 0.10,
    }

    # Default weights for photos WITHOUT faces
    WEIGHTS_NO_FACES = {
        "sharpness": 0.35,
        "exposure": 0.25,
        "composition": 0.25,
        "color": 0.15,
    }

    def __init__(self):
        self.face_analyzer = FaceAnalyzer()

    def analyze_photo(self, image_bytes: bytes, photo_id: str) -> dict:
        """
        Analyze photo quality.

        Args:
            image_bytes: Raw image file bytes
            photo_id: Photo ID for reference

        Returns:
            Dictionary with quality scores
        """
        try:
            # Load image
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != "RGB":
                img = img.convert("RGB")

            # Convert to numpy array for OpenCV
            img_rgb = np.array(img)
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

            # Calculate individual scores
            sharpness_score = self._calculate_sharpness(gray)
            exposure_score = self._calculate_exposure(gray)
            composition_score = self._calculate_composition(gray)
            color_score = self._calculate_color(img_rgb)

            # Face analysis
            face_data = self.face_analyzer.analyze(img_rgb)

            # Calculate overall score based on whether faces are present
            if face_data["has_faces"]:
                weights = self.WEIGHTS_WITH_FACES
                overall_score = (
                    sharpness_score * weights["sharpness"] +
                    face_data["face_quality_score"] * weights["face_quality"] +
                    exposure_score * weights["exposure"] +
                    composition_score * weights["composition"] +
                    color_score * weights["color"]
                )
            else:
                weights = self.WEIGHTS_NO_FACES
                overall_score = (
                    sharpness_score * weights["sharpness"] +
                    exposure_score * weights["exposure"] +
                    composition_score * weights["composition"] +
                    color_score * weights["color"]
                )

            # Load thresholds
            thresholds = self._get_thresholds()
            is_good = overall_score >= thresholds["min_overall"]

            return {
                "overall_score": float(round(overall_score, 1)),
                "is_good_photo": bool(is_good),
                "sharpness_score": float(round(sharpness_score, 1)),
                "exposure_score": float(round(exposure_score, 1)),
                "composition_score": float(round(composition_score, 1)),
                "color_score": float(round(color_score, 1)),
                **face_data,
                "analyzed_at": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"Error analyzing photo {photo_id}: {e}")
            return {
                "overall_score": 0.0,
                "is_good_photo": False,
                "sharpness_score": 0.0,
                "exposure_score": 0.0,
                "composition_score": 0.0,
                "color_score": 0.0,
                "has_faces": False,
                "face_count": 0,
                "face_quality_score": 0.0,
                "best_eyes_open": 0.0,
                "best_expression": 0.0,
                "analyzed_at": datetime.now().isoformat(),
                "error": str(e),
            }

    def _calculate_sharpness(self, gray: np.ndarray) -> float:
        """
        Calculate sharpness score using Laplacian variance.

        Sharp images have high variance (> 500), blurry images have low variance (< 100).
        """
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = float(laplacian.var())

        # Normalize to 0-100 scale
        # < 50 = very blurry (0-20)
        # 50-200 = somewhat blurry (20-50)
        # 200-500 = acceptable (50-75)
        # > 500 = sharp (75-100)
        if variance < 50:
            score = (variance / 50) * 20
        elif variance < 200:
            score = 20 + ((variance - 50) / 150) * 30
        elif variance < 500:
            score = 50 + ((variance - 200) / 300) * 25
        else:
            score = 75 + min(25, ((variance - 500) / 1000) * 25)

        return float(min(100, max(0, score)))

    def _calculate_exposure(self, gray: np.ndarray) -> float:
        """
        Calculate exposure score using histogram analysis.

        Good exposure has minimal clipping on dark/bright ends.
        """
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten()
        total_pixels = float(hist.sum())

        if total_pixels == 0:
            return 50.0

        # Check for clipping
        dark_pct = float(hist[:20].sum()) / total_pixels  # Underexposed pixels
        bright_pct = float(hist[235:].sum()) / total_pixels  # Overexposed pixels

        # Calculate mean brightness
        mean_brightness = float(np.average(np.arange(256), weights=hist))

        # Ideal mean is around 127 (middle)
        brightness_score = 100 - abs(mean_brightness - 127) / 1.27

        # Penalize for clipping
        clipping_penalty = (dark_pct + bright_pct) * 200  # Up to 40 points penalty

        score = brightness_score - clipping_penalty
        return float(min(100, max(0, score)))

    def _calculate_composition(self, gray: np.ndarray) -> float:
        """
        Calculate composition score using edge detection and rule of thirds.

        Good composition has edges/subjects aligned with thirds lines.
        """
        # Detect edges
        edges = cv2.Canny(gray, 100, 200)

        h, w = gray.shape

        # Define rule of thirds zones
        third_w = w // 3
        third_h = h // 3

        # Count edge pixels in each third
        edge_counts = []
        for i in range(3):
            for j in range(3):
                zone = edges[i * third_h:(i + 1) * third_h,
                            j * third_w:(j + 1) * third_w]
                edge_counts.append(float(zone.sum()))

        total_edges = sum(edge_counts)
        if total_edges == 0:
            return 50.0  # Neutral score for images with few edges

        # Ideal: subject not dead center (zone 4)
        center_ratio = edge_counts[4] / total_edges

        # Ideal: some activity along the thirds lines (zones 1, 3, 5, 7)
        thirds_zones = [edge_counts[i] for i in [1, 3, 5, 7]]
        thirds_ratio = sum(thirds_zones) / total_edges

        # Scoring:
        # - Penalize if > 50% of edges are in center
        # - Reward if significant edges are on thirds lines
        center_penalty = max(0, (center_ratio - 0.3) * 100)
        thirds_bonus = thirds_ratio * 50

        score = 50 - center_penalty + thirds_bonus
        return float(min(100, max(0, score)))

    def _calculate_color(self, img_rgb: np.ndarray) -> float:
        """
        Calculate color quality score based on variance and saturation.

        Higher variance = more visually interesting.
        """
        # Convert to HSV
        img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        # Get saturation channel
        saturation = img_hsv[:, :, 1]
        mean_saturation = float(saturation.mean())

        # Get color variance
        color_std = float(img_rgb.std())

        # Score based on saturation (ideal: 50-150 mean)
        if mean_saturation < 30:
            sat_score = mean_saturation / 30 * 50  # Low saturation
        elif mean_saturation < 150:
            sat_score = 50 + ((mean_saturation - 30) / 120) * 40  # Good range
        else:
            sat_score = 90 - ((mean_saturation - 150) / 100) * 20  # Oversaturated

        # Score based on variance (higher = more interesting)
        var_score = min(50, color_std / 2)

        score = sat_score * 0.6 + var_score * 0.4 + 20  # Base boost
        return float(min(100, max(0, score)))

    def _get_thresholds(self) -> dict:
        """Get quality thresholds from config."""
        index = config.load_photo_index()
        quality_config = index.get("quality_config", {})
        return quality_config.get("thresholds", {
            "min_overall": 65,
            "min_sharpness": 50,
            "min_eyes_open": 0.6,
        })

    def close(self):
        """Release resources."""
        self.face_analyzer.close()


class BurstDetector:
    """Detect and group burst photos taken within a short time window."""

    BURST_WINDOW_SECONDS = 2  # Photos within 2 seconds are considered a burst

    def detect_bursts(self, photos: list) -> dict:
        """
        Detect burst groups in a list of photos.

        Args:
            photos: List of photo dicts with client_modified timestamps

        Returns:
            Dictionary mapping burst_group_id to list of photo_ids
        """
        if not photos:
            return {}

        # Sort by timestamp
        sorted_photos = sorted(
            photos,
            key=lambda p: p.get("client_modified", "") or ""
        )

        bursts = {}
        current_burst = []
        burst_counter = 0

        for i, photo in enumerate(sorted_photos):
            timestamp_str = photo.get("client_modified")
            if not timestamp_str:
                continue

            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            except (ValueError, AttributeError):
                continue

            if not current_burst:
                current_burst = [(photo, timestamp)]
                continue

            # Check if within burst window of the last photo
            last_photo, last_timestamp = current_burst[-1]
            time_diff = abs((timestamp - last_timestamp).total_seconds())

            if time_diff <= self.BURST_WINDOW_SECONDS:
                current_burst.append((photo, timestamp))
            else:
                # End of burst - save if more than 1 photo
                if len(current_burst) > 1:
                    burst_id = f"burst_{burst_counter}"
                    bursts[burst_id] = [p["id"] for p, _ in current_burst]
                    burst_counter += 1

                # Start new potential burst
                current_burst = [(photo, timestamp)]

        # Handle last burst
        if len(current_burst) > 1:
            burst_id = f"burst_{burst_counter}"
            bursts[burst_id] = [p["id"] for p, _ in current_burst]

        return bursts

    def select_best_from_burst(self, burst_photos: list) -> Optional[str]:
        """
        Select the best photo from a burst based on quality scores.

        Args:
            burst_photos: List of photo dicts with quality data

        Returns:
            Photo ID of the best photo in the burst
        """
        if not burst_photos:
            return None

        # Sort by overall quality score, descending
        sorted_by_quality = sorted(
            burst_photos,
            key=lambda p: p.get("quality", {}).get("overall_score", 0),
            reverse=True
        )

        return sorted_by_quality[0]["id"]


class ReferencePhotoLearner:
    """Learn quality thresholds from user's reference photos."""

    def __init__(self):
        self.scorer = QualityScorer()

    def calibrate_from_folder(self, folder_path: str) -> dict:
        """
        Analyze reference photos and calculate thresholds.

        Uses 25th percentile of scores as minimum threshold.

        Args:
            folder_path: Path to folder containing good example photos

        Returns:
            Dictionary with calibrated thresholds
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        # Find all images recursively
        image_extensions = {".jpg", ".jpeg", ".png", ".heic"}
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.rglob(f"*{ext}"))
            image_files.extend(folder.rglob(f"*{ext.upper()}"))

        if not image_files:
            raise ValueError(f"No images found in {folder_path}")

        print(f"Analyzing {len(image_files)} reference photos...")

        scores = {
            "overall": [],
            "sharpness": [],
            "exposure": [],
            "composition": [],
            "color": [],
            "face_quality": [],
        }

        for i, img_path in enumerate(image_files):
            try:
                with open(img_path, "rb") as f:
                    image_bytes = f.read()

                result = self.scorer.analyze_photo(image_bytes, str(img_path))

                scores["overall"].append(result["overall_score"])
                scores["sharpness"].append(result["sharpness_score"])
                scores["exposure"].append(result["exposure_score"])
                scores["composition"].append(result["composition_score"])
                scores["color"].append(result["color_score"])
                if result.get("has_faces"):
                    scores["face_quality"].append(result["face_quality_score"])

                if (i + 1) % 10 == 0:
                    print(f"  Analyzed {i + 1}/{len(image_files)} photos...")

            except Exception as e:
                print(f"  Error analyzing {img_path}: {e}")
                continue

        if not scores["overall"]:
            raise ValueError("No photos could be analyzed")

        # Calculate 25th percentile as threshold
        def percentile_25(values):
            if not values:
                return 0
            sorted_values = sorted(values)
            idx = max(0, int(len(sorted_values) * 0.25) - 1)
            return sorted_values[idx]

        thresholds = {
            "min_overall": round(percentile_25(scores["overall"]), 1),
            "min_sharpness": round(percentile_25(scores["sharpness"]), 1),
            "min_eyes_open": 0.6,  # Keep default for eyes
        }

        # Calculate stats for reporting
        stats = {
            "photos_analyzed": len(scores["overall"]),
            "score_distribution": {
                "overall": {
                    "min": round(min(scores["overall"]), 1),
                    "max": round(max(scores["overall"]), 1),
                    "mean": round(sum(scores["overall"]) / len(scores["overall"]), 1),
                    "threshold": thresholds["min_overall"],
                },
                "sharpness": {
                    "min": round(min(scores["sharpness"]), 1),
                    "max": round(max(scores["sharpness"]), 1),
                    "mean": round(sum(scores["sharpness"]) / len(scores["sharpness"]), 1),
                    "threshold": thresholds["min_sharpness"],
                },
            },
        }

        print(f"\nCalibration complete:")
        print(f"  Photos analyzed: {stats['photos_analyzed']}")
        print(f"  Overall score threshold: {thresholds['min_overall']}")
        print(f"  Sharpness threshold: {thresholds['min_sharpness']}")

        return {
            "thresholds": thresholds,
            "stats": stats,
        }

    def save_calibration(self, folder_path: str, thresholds: dict):
        """Save calibration results to the photo index."""
        index = config.load_photo_index()
        index["quality_config"] = {
            "reference_folder": folder_path,
            "thresholds": thresholds,
            "calibrated_at": datetime.now().isoformat(),
        }
        config.save_photo_index(index)

    def close(self):
        """Release resources."""
        self.scorer.close()


# Singleton instances for reuse
_quality_scorer = None


def get_quality_scorer() -> QualityScorer:
    """Get or create singleton QualityScorer instance."""
    global _quality_scorer
    if _quality_scorer is None:
        _quality_scorer = QualityScorer()
    return _quality_scorer


def analyze_photo(image_bytes: bytes, photo_id: str) -> dict:
    """
    Analyze a photo's quality.

    Convenience function using the singleton scorer.

    Args:
        image_bytes: Raw image file bytes
        photo_id: Photo ID for reference

    Returns:
        Dictionary with quality scores
    """
    scorer = get_quality_scorer()
    return scorer.analyze_photo(image_bytes, photo_id)
