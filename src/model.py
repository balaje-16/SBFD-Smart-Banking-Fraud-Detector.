"""Isolation Forest machine learning model wrapper for anomaly detection.

Isolation Forest is an unsupervised ensemble tree algorithm that isolates
anomalies by randomly selecting a feature and randomly selecting a split value.
Because anomalous observations require significantly fewer splits to isolate,
they appear close to the root of the trees.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class IsolationForestAnomalyDetector:
    """Wrapper around scikit-learn's IsolationForest for banking transaction anomaly detection."""

    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 150,
        max_samples: Any = "auto",
        random_state: int = 42,
    ):
        """Initialize Isolation Forest detector.

        Args:
            contamination: Expected proportion of outliers in the dataset.
            n_estimators: Number of isolation trees in ensemble.
            max_samples: Number of samples to draw from X to train each tree.
            random_state: Random state for deterministic reproduction.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.model: Optional[IsolationForest] = None
        self.feature_names: List[str] = []
        self.is_fitted: bool = False
        self.training_metadata: Dict[str, Any] = {}

    def fit(
        self,
        X: np.ndarray,
        feature_names: List[str],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> IsolationForestAnomalyDetector:
        """Fits the Isolation Forest on extracted numerical feature matrix.

        Note: Unsupervised algorithm. Target labels (e.g. is_fraud) MUST NOT be included.

        Args:
            X: 2D numpy array of shape (n_samples, n_features).
            feature_names: List of column names corresponding to columns of X.
            metadata: Optional dictionary with training run information.

        Returns:
            self.
        """
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError(f"Invalid feature matrix shape: {X.shape}. Must be 2D with at least 1 sample.")

        if len(feature_names) != X.shape[1]:
            raise ValueError(
                f"Feature names count ({len(feature_names)}) does not match matrix columns ({X.shape[1]})."
            )

        self.feature_names = list(feature_names)

        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.model.fit(X)
        self.is_fitted = True

        raw_scores = self.model.decision_function(X)
        self.training_metadata = {
            "n_samples": int(X.shape[0]),
            "n_features": int(X.shape[1]),
            "feature_names": self.feature_names,
            "contamination": float(self.contamination),
            "score_min": float(np.min(raw_scores)),
            "score_max": float(np.max(raw_scores)),
            "score_mean": float(np.mean(raw_scores)),
            "score_std": float(np.std(raw_scores)),
            "custom_meta": metadata or {},
        }

        return self

    def predict_raw_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes raw Isolation Forest decision scores.

        In scikit-learn:
        - Negative scores represent anomalies.
        - Positive scores represent normal instances.

        Args:
            X: 2D numpy array of shape (n_samples, n_features).

        Returns:
            1D numpy array of raw decision scores.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model has not been trained yet. Please call fit() before making predictions.")

        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"Feature mismatch: Expected {len(self.feature_names)} features, got {X.shape[1]}."
            )

        # Scikit-learn decision_function: higher = normal, lower = anomalous
        return self.model.decision_function(X)

    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """Predicts binary anomaly labels: +1 for normal, -1 for anomaly."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model is untrained.")
        return self.model.predict(X)

    def save(self, file_path: str) -> None:
        """Serializes fitted model, feature names, and metadata to disk."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Cannot save an untrained model.")

        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        payload = {
            "model": self.model,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "training_metadata": self.training_metadata,
        }
        joblib.dump(payload, file_path)

    @classmethod
    def load(cls, file_path: str) -> IsolationForestAnomalyDetector:
        """Deserializes model checkpoint from disk."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Model checkpoint not found at: {file_path}")

        payload = joblib.load(file_path)
        detector = cls(
            contamination=payload["contamination"],
            n_estimators=payload["n_estimators"],
        )
        detector.model = payload["model"]
        detector.feature_names = payload["feature_names"]
        detector.training_metadata = payload["training_metadata"]
        detector.is_fitted = True

        return detector
