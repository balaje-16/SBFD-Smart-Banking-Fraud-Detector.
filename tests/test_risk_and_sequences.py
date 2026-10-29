"""Tests for anomaly normalization, contextual risk scoring, and sequence detection."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.anomaly_scoring import AnomalyScoreNormalizer
from src.risk_engine import ContextualRiskEngine, RiskWeights
from src.sequence_detection import SequenceDetector


def test_anomaly_score_normalizer():
    normalizer = AnomalyScoreNormalizer()
    raw_scores = np.array([-0.25, -0.10, 0.0, 0.15, 0.20])

    normalizer.calibrate(raw_scores)
    norm = normalizer.normalize(raw_scores)

    # Check bounds
    assert np.all(norm >= 0.0)
    assert np.all(norm <= 100.0)

    # Monotonicity: lowest raw score (-0.25) should have highest normalized score
    assert norm[0] > norm[-1]


def test_risk_weights_validation():
    # Valid weights
    w = RiskWeights(ml_anomaly=0.35, historical_deviation=0.25, velocity=0.20, temporal_timing=0.10, sequence_violation=0.10)
    w.validate()

    # Invalid weights sum
    bad_w = RiskWeights(ml_anomaly=0.8, historical_deviation=0.8)
    with pytest.raises(ValueError, match="must sum to 1.0"):
        bad_w.validate()


def test_contextual_risk_engine():
    engine = ContextualRiskEngine(threshold_medium=40.0, threshold_high=70.0)

    # Test normal transaction
    res_low = engine.evaluate_transaction(
        transaction_id="TX_LOW",
        account_id="ACC_1",
        amount=25.0,
        anomaly_score=10.0,
        deviation_severity=5.0,
        amount_ratio=1.1,
    )
    assert res_low.risk_category == "Low Risk"
    assert res_low.overall_risk_score < 40.0
    assert "Approve" in res_low.recommended_action

    # Test severe composite anomaly
    res_high = engine.evaluate_transaction(
        transaction_id="TX_HIGH",
        account_id="ACC_2",
        amount=9500.0,
        anomaly_score=92.0,
        deviation_severity=85.0,
        amount_ratio=8.5,
        tx_velocity=45.0,
        freq_1h=4.0,
        is_night=True,
        sequence_violation_score=80.0,
    )
    assert res_high.risk_category == "High Risk"
    assert res_high.overall_risk_score >= 70.0
    assert "Block" in res_high.recommended_action
    assert len(res_high.reasons) > 1


def test_sequence_detector_burst():
    base = datetime(2026, 1, 1, 10, 0, 0)
    # 4 transactions within 6 minutes
    df = pd.DataFrame({
        "transaction_id": ["TX_B1", "TX_B2", "TX_B3", "TX_B4"],
        "account_id": ["ACC_BURST", "ACC_BURST", "ACC_BURST", "ACC_BURST"],
        "timestamp": [
            base,
            base + timedelta(minutes=2),
            base + timedelta(minutes=4),
            base + timedelta(minutes=6),
        ],
        "amount": [10.0, 15.0, 12.0, 14.0],
    })

    detector = SequenceDetector(burst_time_window_minutes=15, burst_count_threshold=3)
    violations = detector.detect_transaction_sequences(df)

    assert len(violations) >= 1
    burst_v = next(v for v in violations if v.rule_name == "RAPID_TRANSACTION_BURST")
    assert burst_v.account_id == "ACC_BURST"
    assert len(burst_v.involved_transaction_ids) >= 3


def test_sequence_detector_rapid_drain():
    base = datetime(2026, 1, 1, 10, 0, 0)
    # Account average is $100. Two transactions of $1,000 within 10 minutes.
    df = pd.DataFrame({
        "transaction_id": ["TX_D1", "TX_D2"],
        "account_id": ["ACC_DRAIN", "ACC_DRAIN"],
        "timestamp": [base, base + timedelta(minutes=8)],
        "amount": [1200.0, 1500.0],
    })

    detector = SequenceDetector(drain_transfer_window_minutes=30, drain_count_threshold=2, drain_amount_multiplier=3.0)
    violations = detector.detect_transaction_sequences(df, account_baselines={"ACC_DRAIN": 100.0})

    drain_v = next((v for v in violations if v.rule_name == "RAPID_ACCOUNT_DRAIN"), None)
    assert drain_v is not None
    assert drain_v.account_id == "ACC_DRAIN"


def test_sequence_detector_dormancy():
    base = datetime(2026, 1, 1, 10, 0, 0)
    # First transaction on Jan 1, second on Jan 20 (19 days later) with large amount
    df = pd.DataFrame({
        "transaction_id": ["TX_OLD", "TX_NEW"],
        "account_id": ["ACC_DORM", "ACC_DORM"],
        "timestamp": [base, base + timedelta(days=19)],
        "amount": [50.0, 800.0],
    })

    detector = SequenceDetector(dormancy_days_threshold=14)
    violations = detector.detect_transaction_sequences(df, account_baselines={"ACC_DORM": 50.0})

    dorm_v = next((v for v in violations if v.rule_name == "DORMANT_ACCOUNT_SUDDEN_SPIKE"), None)
    assert dorm_v is not None
    assert dorm_v.account_id == "ACC_DORM"


def test_correlate_auth_events():
    df_tx = pd.DataFrame({
        "transaction_id": ["TX_HACK"],
        "account_id": ["ACC_VICTIM"],
        "timestamp": ["2026-01-15 02:25:00"],
        "amount": [5000.0],
    })
    df_auth = pd.DataFrame({
        "event_id": ["AUTH_1"],
        "account_id": ["ACC_VICTIM"],
        "timestamp": ["2026-01-15 02:15:00"],
        "event_type": ["LOGIN_FAILED"],
        "ip_address": ["194.26.29.112"],
    })

    correlated = SequenceDetector.correlate_auth_events(df_tx, df_auth, window_minutes=30)
    assert len(correlated) == 1
    assert correlated[0]["transaction_id"] == "TX_HACK"
    assert correlated[0]["account_id"] == "ACC_VICTIM"
