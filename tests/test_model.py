"""Tests for Isolation Forest model encapsulation, inference, and persistence."""

import os
import pytest
import numpy as np

from src.model import IsolationForestAnomalyDetector


def test_model_training_and_inference(tmp_path):
    rng = np.random.default_rng(42)
    # Generate 100 normal points clustered near (0, 0) and 5 outlier points near (10, 10)
    normal = rng.normal(0, 1, size=(100, 3))
    outliers = rng.uniform(8, 12, size=(5, 3))
    X = np.vstack([normal, outliers])
    feature_names = ["feat_1", "feat_2", "feat_3"]

    model = IsolationForestAnomalyDetector(contamination=0.05, random_state=42)
    assert not model.is_fitted

    # Untrained model raises error
    with pytest.raises(RuntimeError, match="untrained|not been trained"):
        model.predict_raw_scores(X)

    # Train model
    model.fit(X, feature_names)
    assert model.is_fitted

    raw_scores = model.predict_raw_scores(X)
    assert len(raw_scores) == 105

    # Outliers should have noticeably lower raw decision scores than normal points
    normal_scores = raw_scores[:100]
    outlier_scores = raw_scores[100:]
    assert np.mean(outlier_scores) < np.mean(normal_scores)

    # Save and reload
    save_file = os.path.join(tmp_path, "test_model.joblib")
    model.save(save_file)
    assert os.path.exists(save_file)

    loaded_model = IsolationForestAnomalyDetector.load(save_file)
    assert loaded_model.is_fitted
    assert loaded_model.feature_names == feature_names

    reloaded_scores = loaded_model.predict_raw_scores(X)
    assert np.allclose(raw_scores, reloaded_scores)


def test_model_feature_mismatch_raises():
    X = np.zeros((10, 3))
    model = IsolationForestAnomalyDetector()
    model.fit(X, ["a", "b", "c"])

    # Predict with 2 features instead of 3
    with pytest.raises(ValueError, match="Feature mismatch"):
        model.predict_raw_scores(np.zeros((5, 2)))
