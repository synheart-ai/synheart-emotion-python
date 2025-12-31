"""Synheart Emotion - On-device emotion inference from biosignals.

This package provides real-time emotion detection from heart rate and
RR interval data, designed for wearable devices and health monitoring apps.
"""

__version__ = "0.0.1"

from .synheart_emotion import (
    EmotionConfig,
    EmotionEngine,
    EmotionResult,
    EmotionError,
    TooFewRRError,
    BadInputError,
    ModelIncompatibleError,
    FeatureExtractionError,
)

__all__ = [
    "EmotionConfig",
    "EmotionEngine",
    "EmotionResult",
    "EmotionError",
    "TooFewRRError",
    "BadInputError",
    "ModelIncompatibleError",
    "FeatureExtractionError",
]
