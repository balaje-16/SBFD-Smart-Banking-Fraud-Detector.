"""Tests for chronological model evaluation, PR-AUC, confusion matrices, and unlabelled reporting."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.evaluation import ModelEvaluator, EvaluationMetrics


def test_chronological_split():
    base = datetime(2026, 1, 1, 0, 0, 0)
    df = pd.DataFrame({
        "transaction_id": [f"TX_{i}" for i in range(100)],
        "timestamp": [base + timedelta(minutes=i * 10) for i in range(100)],
        "amount": [50.0] * 100,
    })

    train_df, test_df = ModelEvaluator.chronological_split(df, test_ratio=0.30)

    assert len(train_df) == 70
    assert len(test_df) == 30

    # Ensure strict temporal separation: all train timestamps < all test timestamps
    assert train_df["timestamp"].max() < test_df["timestamp"].min()


def test_evaluation_metrics_calculation():
    # 90 legitimate, 10 fraud
    # Predictions:
    # 8 true positives (score >= 70)
    # 2 false negatives (fraud with score < 70)
    # 3 false positives (legitimate with score >= 70)
    # 87 true negatives (legitimate with score < 70)
    y_true = np.array([1] * 8 + [1] * 2 + [0] * 3 + [0] * 87)
    scores = np.array([85.0] * 8 + [40.0] * 2 + [75.0] * 3 + [20.0] * 87)

    metrics = ModelEvaluator.evaluate_predictions(y_true, scores, threshold=70.0)

    assert metrics.has_ground_truth is True
    assert metrics.true_positives == 8
    assert metrics.false_negatives == 2
    assert metrics.false_positives == 3
    assert metrics.true_negatives == 87

    # Precision = 8 / (8 + 3) = 8 / 11 = ~0.727
    assert np.isclose(metrics.precision, 8 / 11, atol=1e-3)
    # Recall = 8 / (8 + 2) = 0.8
    assert np.isclose(metrics.recall, 0.8, atol=1e-3)
    # ROC-AUC should be high
    assert metrics.roc_auc is not None and metrics.roc_auc > 0.85
    # PR-AUC should be high
    assert metrics.pr_auc is not None and metrics.pr_auc > 0.60


def test_unlabeled_anomalies_summary():
    scores = np.array([15.0, 25.0, 35.0, 50.0, 65.0, 85.0, 95.0])
    summary = ModelEvaluator.summarize_unlabeled_anomalies(scores)

    assert summary["has_ground_truth"] is False
    assert summary["total_records"] == 7
    assert summary["high_risk_count"] == 2  # 85.0, 95.0
    assert summary["medium_risk_count"] == 2  # 50.0, 65.0
    assert summary["low_risk_count"] == 3  # 15.0, 25.0, 35.0
