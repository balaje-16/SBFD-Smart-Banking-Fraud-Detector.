"""Tests for SOC alert lifecycle, notes, status workflow, and local file persistence."""

import os
import pytest
import pandas as pd

from src.alert_manager import (
    AlertManager,
    FraudAlert,
    ALERT_STATUS_NEW,
    ALERT_STATUS_UNDER_REVIEW,
    ALERT_STATUS_RESOLVED,
    ALERT_STATUS_FALSE_POSITIVE,
)
from src.risk_engine import RiskEvaluationResult


def test_alert_manager_lifecycle(tmp_path):
    storage = os.path.join(tmp_path, "test_alerts.json")
    mgr = AlertManager(storage_path=storage)

    eval_results = [
        RiskEvaluationResult(
            transaction_id="TX_SAFE",
            account_id="ACC_1",
            amount=30.0,
            overall_risk_score=20.0,
            risk_category="Low Risk",
            anomaly_score=15.0,
        ),
        RiskEvaluationResult(
            transaction_id="TX_WARN",
            account_id="ACC_2",
            amount=500.0,
            overall_risk_score=55.0,
            risk_category="Medium Risk",
            anomaly_score=60.0,
            reasons=["High velocity"],
            recommended_action="OTP verification",
        ),
        RiskEvaluationResult(
            transaction_id="TX_ALERT",
            account_id="ACC_3",
            amount=8000.0,
            overall_risk_score=85.0,
            risk_category="High Risk",
            anomaly_score=90.0,
            reasons=["Outlier amount", "Night execution"],
            recommended_action="Block transfer",
        ),
    ]

    # Generate alerts: Low risk should be skipped; Medium and High should create alerts
    created = mgr.generate_alerts_from_evaluations(eval_results)
    assert len(created) == 2
    assert os.path.exists(storage)

    # Check dataframe export
    df_alerts = mgr.to_dataframe()
    assert len(df_alerts) == 2
    assert "TX_ALERT" in df_alerts["transaction_id"].values
    assert "TX_WARN" in df_alerts["transaction_id"].values

    # Test status update and analyst notes
    high_alert = next(a for a in created if a.transaction_id == "TX_ALERT")
    alert_id = high_alert.alert_id

    mgr.update_alert_status(
        alert_id=alert_id,
        new_status=ALERT_STATUS_UNDER_REVIEW,
        analyst_notes="Spoke with branch manager. Pending customer callback.",
    )

    # Verify update persisted by creating a new AlertManager instance from same storage file
    mgr2 = AlertManager(storage_path=storage)
    loaded = mgr2.alerts[alert_id]
    assert loaded.status == ALERT_STATUS_UNDER_REVIEW
    assert "Pending customer callback" in loaded.analyst_notes


def test_alert_manager_invalid_status_raises(tmp_path):
    storage = os.path.join(tmp_path, "test_alerts_err.json")
    mgr = AlertManager(storage_path=storage)
    alert = FraudAlert(
        alert_id="ALT_999",
        transaction_id="TX_1",
        account_id="A1",
        risk_score=75.0,
        risk_category="High Risk",
        detection_timestamp="2026-01-01 10:00:00",
        reasons=[],
        recommended_action="",
    )
    mgr.alerts[alert.alert_id] = alert

    with pytest.raises(ValueError, match="Invalid status"):
        mgr.update_alert_status("ALT_999", "INVALID_STATUS_XYZ")
