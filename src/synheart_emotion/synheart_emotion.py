"""Synheart Emotion - On-device emotion inference from biosignals.

Single-file implementation with ONNX models, 14-feature HRV extraction, and windowing.
"""

import json
import math
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort
from scipy import signal as sp_signal


# Trapezoidal integration implementation
def _trapz(y: Any, x: Optional[Any] = None) -> float:
    """Simple trapezoidal integration."""
    y = np.asarray(y)
    if x is None:
        dx = 1.0
        return np.sum((y[1:] + y[:-1]) / 2.0) * dx
    x = np.asarray(x)
    if len(x) != len(y):
        raise ValueError("x and y must have same length")
    if len(y) < 2:
        return 0.0
    return np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(x))


# ============================================================================
# HRV FEATURE EXTRACTION (14 features)
# ============================================================================


def _clean_rr_intervals(rr_intervals_ms: List[float]) -> List[float]:
    """Clean RR intervals by removing physiologically invalid values."""
    MIN_VALID_RR_MS = 300.0  # 200 BPM
    MAX_VALID_RR_MS = 2000.0  # 30 BPM
    return [rr for rr in rr_intervals_ms if MIN_VALID_RR_MS <= rr <= MAX_VALID_RR_MS]


def extract_14_features(
    rr_intervals_ms: List[float], mean_hr: Optional[float] = None
) -> List[float]:
    """Extract all 14 HRV features required for ExtraTrees models.

    Returns features in order: ['RMSSD', 'Mean_RR', 'HRV_SDNN', 'pNN50',
    'HRV_HF', 'HRV_LF', 'HRV_HF_nu', 'HRV_LF_nu', 'HRV_LFHF', 'HRV_TP',
    'HRV_SD1SD2', 'HRV_Sampen', 'HRV_DFA_alpha1', 'HR']
    """
    valid_rr = _clean_rr_intervals(rr_intervals_ms)
    if len(valid_rr) < 10:
        return [0.0] * 14

    # Time-domain features
    valid_arr = np.array(valid_rr)
    diffs = np.diff(valid_arr)
    rmssd = float(np.sqrt(np.mean(diffs**2)))
    mean_rr = float(np.mean(valid_arr))
    sdnn = float(np.std(valid_arr, ddof=1))
    pnn50 = float((np.sum(np.abs(diffs) > 50) / len(diffs)) * 100.0) if len(diffs) > 0 else 0.0

    # Frequency-domain features
    rr_sec = valid_arr / 1000.0
    time_points = np.cumsum(np.concatenate([[0], rr_sec[:-1]]))
    sampling_rate = 4.0
    uniform_time = np.arange(0, time_points[-1], 1.0 / sampling_rate)

    if len(uniform_time) >= 16:
        interpolated_rr = np.interp(uniform_time, time_points, rr_sec)
        interpolated_rr = interpolated_rr - np.mean(interpolated_rr)
        freqs, psd = sp_signal.welch(
            interpolated_rr, fs=sampling_rate, nperseg=min(256, len(interpolated_rr))
        )

        vlf_idx = np.logical_and(freqs >= 0.0033, freqs <= 0.04)
        lf_idx = np.logical_and(freqs >= 0.04, freqs <= 0.15)
        hf_idx = np.logical_and(freqs >= 0.15, freqs <= 0.4)

        vlf = float(_trapz(psd[vlf_idx], freqs[vlf_idx]) if np.any(vlf_idx) else 0.0)
        lf = float(_trapz(psd[lf_idx], freqs[lf_idx]) if np.any(lf_idx) else 0.0)
        hf = float(_trapz(psd[hf_idx], freqs[hf_idx]) if np.any(hf_idx) else 0.0)
        tp = vlf + lf + hf

        lfhf_sum = lf + hf
        lf_nu = float((lf / lfhf_sum) if lfhf_sum > 0 else 0.0)
        hf_nu = float((hf / lfhf_sum) if lfhf_sum > 0 else 0.0)
        lfhf = float((lf / hf) if hf > 0 else 0.0)
    else:
        hf = lf = lf_nu = hf_nu = lfhf = tp = 0.0

    # Non-linear features
    if len(valid_arr) >= 3:
        var_diff = np.var(diffs, ddof=1)
        sd1 = np.sqrt(0.5 * var_diff)
        sd2_squared = 2 * (sdnn**2) - 0.5 * (sd1**2)
        sd2 = np.sqrt(sd2_squared) if sd2_squared > 0 else 0.0
        sd1sd2 = float((sd1 / sd2) if sd2 > 0 else 0.0)

        # Sample Entropy (simplified)
        tolerance = 0.2 * sdnn
        if tolerance > 0 and len(valid_arr) >= 4:
            m = 2
            count_m = sum(
                1
                for i in range(len(valid_arr) - m)
                for j in range(i + 1, len(valid_arr) - m)
                if all(abs(valid_arr[i + k] - valid_arr[j + k]) <= tolerance for k in range(m))
            )
            count_m1 = sum(
                1
                for i in range(len(valid_arr) - m - 1)
                for j in range(i + 1, len(valid_arr) - m - 1)
                if all(abs(valid_arr[i + k] - valid_arr[j + k]) <= tolerance for k in range(m + 1))
            )
            samp_en = float(-math.log(count_m1 / count_m)) if count_m > 0 and count_m1 > 0 else 0.0
        else:
            samp_en = 0.0

        # DFA alpha1 (simplified)
        if len(valid_arr) >= 16:
            cum_sum = np.cumsum(valid_arr - mean_rr)
            box_sizes = [4, 6, 8, 10, 12, 14, 16]
            log_sizes, log_fluct = [], []
            for n in box_sizes:
                if n > len(cum_sum):
                    break
                num_boxes = len(cum_sum) // n
                if num_boxes < 1:
                    continue
                total_var = 0.0
                for i in range(num_boxes):
                    seg = cum_sum[i * n : (i + 1) * n]
                    x = np.arange(n)
                    coeffs = np.polyfit(x, seg, 1)
                    trend = np.polyval(coeffs, x)
                    total_var += np.mean((seg - trend) ** 2)
                fluct = np.sqrt(total_var / num_boxes)
                if fluct > 0:
                    log_sizes.append(math.log(n))
                    log_fluct.append(math.log(fluct))
            dfa_alpha1 = (
                float(np.polyfit(log_sizes, log_fluct, 1)[0]) if len(log_sizes) >= 2 else 0.0
            )
        else:
            dfa_alpha1 = 0.0
    else:
        sd1sd2 = samp_en = dfa_alpha1 = 0.0

    # Heart rate
    hr = (
        float(mean_hr)
        if mean_hr is not None
        else (float((60000.0 / mean_rr) if mean_rr > 0 else 0.0))
    )

    return [
        rmssd,
        mean_rr,
        sdnn,
        pnn50,
        hf,
        lf,
        hf_nu,
        lf_nu,
        lfhf,
        tp,
        sd1sd2,
        samp_en,
        dfa_alpha1,
        hr,
    ]


# ============================================================================
# ONNX MODEL LOADING
# ============================================================================


def _load_model_from_package(
    model_name: str,
) -> Tuple[ort.InferenceSession, List[str], List[str], Dict[str, Any]]:
    """Load ONNX model from package data directory."""
    try:
        import synheart_emotion

        package_path = Path(synheart_emotion.__file__).parent
    except (ImportError, AttributeError):
        # Fallback: try to get path from __file__ if running as module
        current_file = Path(__file__)
        package_path = current_file.parent

    data_dir = package_path / "data"
    model_path = data_dir / f"{model_name}.onnx"

    # Try different metadata naming patterns
    parts = model_name.replace(".onnx", "").split("_")
    metadata_path = None
    if len(parts) >= 4:
        meta_name = f"{parts[0]}_metadata_{parts[1]}_{parts[2]}_{'_'.join(parts[3:])}.json"
        metadata_path = data_dir / meta_name
    else:
        metadata_path = data_dir / f"{model_name}_metadata.json"

    if not metadata_path.exists() or not model_path.exists():
        raise ValueError(f"Model files not found: {model_path}, {metadata_path}")

    # Load metadata
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    # Load ONNX model
    session = ort.InferenceSession(str(model_path))
    input_names = metadata["schema"]["input_names"]
    class_names = metadata["output"]["class_names"]

    return session, input_names, class_names, metadata


# ============================================================================
# MAIN ENGINE
# ============================================================================


@dataclass
class EmotionConfig:
    """Configuration for emotion inference."""

    model_id: str = "extratrees_w120s60_binary_v1_0"
    window_seconds: float = 120.0
    step_seconds: float = 60.0
    min_rr_count: int = 30
    return_all_probas: bool = True
    hr_baseline: Optional[float] = None
    priors: Optional[Dict[str, float]] = None


class EmotionEngine:
    """Emotion inference engine with windowing and ONNX models."""

    @classmethod
    def from_pretrained(
        cls,
        config: EmotionConfig,
        model: Optional[Any] = None,
        on_log: Optional[Callable[[str, str, Optional[Dict[str, Any]]], None]] = None,
    ) -> "EmotionEngine":
        """Create engine from pretrained model."""
        return cls(config=config, on_log=on_log)

    def __init__(
        self,
        config: EmotionConfig,
        on_log: Optional[Callable[[str, str, Optional[Dict[str, Any]]], None]] = None,
    ):
        self.config = config
        self.on_log = on_log
        self._buffer: deque = deque()
        self._last_emission: Optional[datetime] = None
        self._lock = threading.RLock()

        # Load model
        try:
            model_name_map = {
                "extratrees_w120s60_binary_v1_0": "ExtraTrees_120_60_nozipmap",
                "extratrees_w120s5_binary_v1_0": "ExtraTrees_120_5_nozipmap",
                "extratrees_w60s5_binary_v1_0": "ExtraTrees_60_5_nozipmap",
            }
            model_name = model_name_map.get(config.model_id, config.model_id)
            if model_name.endswith(".onnx"):
                model_name = model_name[:-5]

            self._session, self._input_names, self._class_names, self._metadata = (
                _load_model_from_package(model_name)
            )
            self._input_name = self._session.get_inputs()[0].name
        except Exception as e:
            raise ValueError(f"Failed to load model '{config.model_id}': {e}") from e

    def push(
        self,
        hr: float,
        rr_intervals_ms: List[float],
        timestamp: datetime,
        motion: Optional[Dict[str, float]] = None,
    ) -> None:
        """Push new data point into the engine."""
        with self._lock:
            if hr < 30.0 or hr > 300.0:
                self._log("warn", f"Invalid HR value: {hr}")
                return
            if not rr_intervals_ms:
                self._log("warn", "Empty RR intervals")
                return

            self._buffer.append(
                {
                    "timestamp": timestamp,
                    "hr": hr,
                    "rr_intervals_ms": list(rr_intervals_ms),
                    "motion": motion,
                }
            )

            # Trim old data
            cutoff_time = datetime.now() - timedelta(seconds=self.config.window_seconds)
            while self._buffer and self._buffer[0]["timestamp"] < cutoff_time:
                self._buffer.popleft()

            self._log("debug", f"Pushed: HR={hr}, RR count={len(rr_intervals_ms)}")

    def consume_ready(self) -> List["EmotionResult"]:
        """Consume ready results (throttled by step interval)."""
        results: List["EmotionResult"] = []
        with self._lock:
            now = datetime.now()
            if self._last_emission is not None:
                elapsed = (now - self._last_emission).total_seconds()
                if elapsed < self.config.step_seconds:
                    return results

            if len(self._buffer) < 2:
                return results

            # Check window is full
            oldest_age = (now - self._buffer[0]["timestamp"]).total_seconds()
            if oldest_age < (self.config.window_seconds - 2.0):
                self._log(
                    "warn",
                    f"Window insufficient: {oldest_age:.1f}s < {self.config.window_seconds}s",
                )
                return results

            # Collect data
            hr_values = []
            all_rr = []
            for point in self._buffer:
                hr_values.append(point["hr"])
                all_rr.extend(point["rr_intervals_ms"])

            if len(all_rr) < self.config.min_rr_count:
                self._log(
                    "warn", f"Too few RR intervals: {len(all_rr)} < {self.config.min_rr_count}"
                )
                return results

            # Extract features
            mean_hr: Optional[float] = (
                float(np.mean(hr_values)) if hr_values else None
            )
            feature_list = extract_14_features(all_rr, mean_hr=mean_hr)

            # Map to feature names
            feature_names = [
                "RMSSD",
                "Mean_RR",
                "HRV_SDNN",
                "pNN50",
                "HRV_HF",
                "HRV_LF",
                "HRV_HF_nu",
                "HRV_LF_nu",
                "HRV_LFHF",
                "HRV_TP",
                "HRV_SD1SD2",
                "HRV_Sampen",
                "HRV_DFA_alpha1",
                "HR",
            ]
            features = dict(zip(feature_names, feature_list))

            # Run inference
            input_vector = [features[name] for name in self._input_names]
            input_data = np.array(input_vector, dtype=np.float32).reshape(1, -1)
            outputs = self._session.run(None, {self._input_name: input_data})

            # Parse outputs
            prob_output = outputs[1]  # Second output is probabilities
            if prob_output.ndim == 2:
                probabilities = prob_output[0]
            else:
                probabilities = prob_output

            # Normalize
            prob_sum: float = float(np.sum(probabilities))
            if prob_sum > 0 and abs(prob_sum - 1.0) > 0.001:
                probabilities = probabilities / (prob_sum + 1e-8)

            # Create result
            prob_dict = {self._class_names[i]: float(prob) for i, prob in enumerate(probabilities)}

            result = EmotionResult.from_inference(
                timestamp=now,
                probabilities=prob_dict,
                features=features,
                model={
                    "id": self._metadata.get("model_id", "unknown"),
                    "version": self._metadata.get("version", "1.0"),
                    "class_names": self._class_names,
                },
            )

            results.append(result)
            self._last_emission = now
            self._log("info", f"Emitted: {result.emotion} ({result.confidence*100:.1f}%)")

        return results

    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get current buffer statistics."""
        with self._lock:
            if not self._buffer:
                return {"count": 0, "duration_ms": 0, "hr_range": [0.0, 0.0], "rr_count": 0}

            hr_values = [p["hr"] for p in self._buffer]
            rr_count = sum(len(p["rr_intervals_ms"]) for p in self._buffer)
            duration = (
                self._buffer[-1]["timestamp"] - self._buffer[0]["timestamp"]
            ).total_seconds() * 1000

            return {
                "count": len(self._buffer),
                "duration_ms": int(duration),
                "hr_range": [min(hr_values), max(hr_values)],
                "rr_count": rr_count,
            }

    def clear(self) -> None:
        """Clear all buffered data."""
        with self._lock:
            self._buffer.clear()
            self._last_emission = None
            self._log("info", "Buffer cleared")

    def _log(self, level: str, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log message."""
        if self.on_log:
            self.on_log(level, message, context)


# ============================================================================
# EMOTION ERRORS
# ============================================================================


class EmotionError(Exception):
    """Base exception class for emotion inference errors."""

    pass


class TooFewRRError(EmotionError):
    """Raised when there are too few RR intervals for inference."""

    pass


class BadInputError(EmotionError):
    """Raised when input data is invalid."""

    pass


class ModelIncompatibleError(EmotionError):
    """Raised when model is incompatible with features."""

    pass


class FeatureExtractionError(EmotionError):
    """Raised when feature extraction fails."""

    pass


# ============================================================================
# EMOTION RESULT
# ============================================================================


@dataclass
class EmotionResult:
    """Result of emotion inference."""

    timestamp: datetime
    emotion: str
    confidence: float
    probabilities: Dict[str, float]
    features: Dict[str, float]
    model: Dict[str, Any]

    @classmethod
    def from_inference(
        cls,
        timestamp: datetime,
        probabilities: Dict[str, float],
        features: Dict[str, float],
        model: Dict[str, Any],
    ) -> "EmotionResult":
        """Create from raw inference data."""
        # Find top-1 emotion
        top_emotion = max(probabilities.items(), key=lambda x: x[1])
        return cls(
            timestamp=timestamp,
            emotion=top_emotion[0],
            confidence=top_emotion[1],
            probabilities=probabilities,
            features=features,
            model=model,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "emotion": self.emotion,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "features": self.features,
            "model": self.model,
        }
