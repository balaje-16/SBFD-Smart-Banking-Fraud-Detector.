"""Anomaly score calibration and normalization module.

Transforms raw Isolation Forest decision values into an intuitive,
normalized 0-100 Anomaly Severity Index where higher values denote greater anomaly.
Clearly documents normalization methodology and guards against interpreting scores as
calibrated Bayesian probabilities.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


class AnomalyScoreNormalizer:
    """Normalizes raw decision function scores from Isolation Forest into a 0-100 scale."""

    def __init__(
        self,
        score_min: Optional[float] = None,
        score_max: Optional[float] = None,
    ):
        """Initialize normalizer with optional pre-calibrated boundaries.

        Args:
            score_min: Lower percentile of raw scores (most anomalous).
            score_max: Upper percentile of raw scores (most normal).
        """
        self.score_min = score_min
        self.score_max = score_max
        self.is_calibrated = (score_min is not None and score_max is not None)

    def calibrate(self, raw_scores: np.ndarray, lower_pct: float = 0.5, upper_pct: float = 99.5) -> None:
        """Calibrates normalization boundaries using training dataset distribution.

        Args:
            raw_scores: Raw output from Isolation Forest decision_function().
            lower_pct: Lower percentile to map near 100 anomaly.
            upper_pct: Upper percentile to map near 0 anomaly.
        """
        if len(raw_scores) == 0:
            raise ValueError("Cannot calibrate normalizer on empty score array.")

        self.score_min = float(np.percentile(raw_scores, lower_pct))
        self.score_max = float(np.percentile(raw_scores, upper_pct))

        # Guard against zero range
        if abs(self.score_max - self.score_min) < 1e-6:
            self.score_max = self.score_min + 0.1

        self.is_calibrated = True

    def normalize(self, raw_scores: np.ndarray) -> np.ndarray:
        """Converts raw decision scores to 0-100 anomaly severity scores.

        Methodology:
        Scikit-learn's `decision_function` returns positive values for normal instances
        and negative values for anomalies.
        The transformation maps:
          raw_score <= score_min -> 100.0 (Extreme Anomaly)
          raw_score >= score_max -> 0.0 (Typical Normal)
        Formula:
          norm_score = 100.0 * (score_max - raw_score) / (score_max - score_min)

        Args:
            raw_scores: Array of raw decision function values.

        Returns:
            Array of normalized scores bounded in [0.0, 100.0].
        """
        if not self.is_calibrated or self.score_min is None or self.score_max is None:
            # Dynamic fallback calibration if not explicitly calibrated
            self.calibrate(raw_scores)

        span = self.score_max - self.score_min
        norm = 100.0 * (self.score_max - raw_scores) / (span + 1e-8)
        return np.clip(norm, 0.0, 100.0)

    def to_dict(self) -> Dict[str, Any]:
        """Export calibration parameters."""
        return {
            "score_min": self.score_min,
            "score_max": self.score_max,
            "is_calibrated": self.is_calibrated,
            "normalization_method": "Percentile-calibrated linear inverted scaling",
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> AnomalyScoreNormalizer:
        """Reconstruct normalizer from dictionary."""
        return cls(
            score_min=data.get("score_min"),
            score_max=data.get("score_max"),
        )
