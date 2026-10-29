"""Contextual Risk Scoring Engine.

Synthesizes multiple orthogonal cyber and behavioural risk signals:
- Machine Learning Anomaly Score (Isolation Forest)
- Account-level Historical Spending Deviation
- Transaction Velocity & Frequency Burst
- Temporal Risk (Night / Off-hours Execution)
- Multi-event Sequence Violations

Returns a composite 0-100 Risk Score, categorical classification
(Low, Medium, High Risk), and human-readable evidence-based explanations.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional
import numpy as np


@dataclass
class RiskWeights:
    """Configurable weights for contextual risk signal fusion. Must sum to 1.0."""

    ml_anomaly: float = 0.35
    historical_deviation: float = 0.25
    velocity: float = 0.20
    temporal_timing: float = 0.10
    sequence_violation: float = 0.10

    def validate(self) -> None:
        """Verify weights sum to approximately 1.0 and are non-negative."""
        total = self.ml_anomaly + self.historical_deviation + self.velocity + self.temporal_timing + self.sequence_violation
        if not (0.98 <= total <= 1.02):
            raise ValueError(f"Risk weights must sum to 1.0 (Current sum: {total:.4f}).")
        for k, v in asdict(self).items():
            if v < 0:
                raise ValueError(f"Weight for '{k}' cannot be negative: {v}")


@dataclass
class RiskEvaluationResult:
    """Complete evaluation outcome for an individual transaction."""

    transaction_id: str
    account_id: Optional[str]
    amount: float
    overall_risk_score: float
    risk_category: str
    anomaly_score: float
    contributing_signals: Dict[str, float] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)
    recommended_action: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return asdict(self)


class ContextualRiskEngine:
    """Evaluates contextual risk and generates explainable cybersecurity justifications."""

    def __init__(
        self,
        weights: Optional[RiskWeights] = None,
        threshold_medium: float = 40.0,
        threshold_high: float = 70.0,
    ):
        """Initialize risk engine.

        Args:
            weights: Optional RiskWeights configuration.
            threshold_medium: Score boundary for Medium Risk (default 40.0).
            threshold_high: Score boundary for High Risk (default 70.0).
        """
        self.weights = weights or RiskWeights()
        self.weights.validate()
        self.threshold_medium = threshold_medium
        self.threshold_high = threshold_high

    def evaluate_transaction(
        self,
        transaction_id: str,
        account_id: Optional[str],
        amount: float,
        anomaly_score: float,
        deviation_severity: float = 0.0,
        amount_ratio: float = 1.0,
        tx_velocity: float = 0.0,
        freq_1h: float = 1.0,
        is_night: bool = False,
        is_weekend: bool = False,
        sequence_violation_score: float = 0.0,
        sequence_notes: Optional[List[str]] = None,
        is_cold_start: bool = False,
    ) -> RiskEvaluationResult:
        """Computes composite risk score and generates evidence-based explanations.

        Args:
            transaction_id: Unique transaction ID.
            account_id: Associated account ID or None.
            amount: Transaction currency amount.
            anomaly_score: 0-100 Isolation Forest severity score.
            deviation_severity: 0-100 score reflecting spending ratio vs account baseline.
            amount_ratio: Multiple of historical average amount.
            tx_velocity: Calculated velocity ($ / second).
            freq_1h: Transaction count in past 1 hour.
            is_night: Whether executed between 23:00 and 05:00.
            is_weekend: Whether executed on Saturday or Sunday.
            sequence_violation_score: 0-100 score from SequenceDetector.
            sequence_notes: Explanations from sequence violations.
            is_cold_start: Whether account lacks sufficient history.

        Returns:
            RiskEvaluationResult object with full audit trail.
        """
        # 1. Normalize individual signals to [0, 100]
        # ML signal
        s_ml = np.clip(float(anomaly_score), 0.0, 100.0)

        # Behavioural historical deviation signal
        s_dev = np.clip(float(deviation_severity), 0.0, 100.0)

        # Velocity & frequency signal (Velocity > $20/s or freq > 3 tx/hr is high risk)
        norm_vel = min(60.0, (tx_velocity / 15.0) * 30.0)
        norm_freq = min(40.0, max(0.0, freq_1h - 1.0) * 15.0)
        s_vel = float(np.clip(norm_vel + norm_freq, 0.0, 100.0))

        # Temporal risk: Off-hours or weekend transfers
        s_time = 0.0
        if is_night:
            s_time += 60.0
        if is_weekend:
            s_time += 25.0
        s_time = float(np.clip(s_time, 0.0, 100.0))

        # Sequence violation signal
        s_seq = np.clip(float(sequence_violation_score), 0.0, 100.0)

        # If cold-start account, slightly dampen historical deviation weight and shift to ML
        w = self.weights
        if is_cold_start:
            effective_w_ml = w.ml_anomaly + (w.historical_deviation * 0.5)
            effective_w_dev = w.historical_deviation * 0.5
        else:
            effective_w_ml = w.ml_anomaly
            effective_w_dev = w.historical_deviation

        # Compute weighted composite score
        overall_score = (
            effective_w_ml * s_ml
            + effective_w_dev * s_dev
            + w.velocity * s_vel
            + w.temporal_timing * s_time
            + w.sequence_violation * s_seq
        )
        overall_score = round(float(np.clip(overall_score, 0.0, 100.0)), 1)

        # Categorize Risk
        if overall_score >= self.threshold_high:
            category = "High Risk"
            recommended_action = "Block transaction immediately; initiate step-up 2FA and analyst contact."
        elif overall_score >= self.threshold_medium:
            category = "Medium Risk"
            recommended_action = "Flag for manual compliance review; request out-of-band SMS/OTP confirmation."
        else:
            category = "Low Risk"
            recommended_action = "Approve and process normally."

        # Compile human-readable diagnostic reasons
        reasons: List[str] = []

        if s_ml >= 70.0:
            reasons.append(f"Machine learning anomaly detector isolated rare multidimensional pattern (Score: {s_ml:.1f}/100).")
        elif s_ml >= 50.0:
            reasons.append(f"Mild anomaly detected by Isolation Forest (Score: {s_ml:.1f}/100).")

        if amount_ratio >= 3.0 and not is_cold_start:
            reasons.append(f"Transaction amount is {amount_ratio:.1f}x higher than this account's historical average.")

        if freq_1h >= 3.0:
            reasons.append(f"High transaction frequency: {int(freq_1h)} transactions recorded within a 1-hour window.")

        if tx_velocity > 15.0:
            reasons.append(f"Elevated velocity rate: ${tx_velocity:.2f}/sec since preceding account activity.")

        if is_night:
            reasons.append("Executed during high-risk overnight hours (23:00 - 05:00).")

        if sequence_notes:
            reasons.extend(sequence_notes)

        if is_cold_start and overall_score >= self.threshold_medium:
            reasons.append("New account with limited historical transactions; evaluated against population baselines.")

        if not reasons:
            reasons.append("Transaction characteristics align with normal operating baselines.")

        contributing = {
            "ml_anomaly_signal": round(s_ml, 1),
            "historical_deviation_signal": round(s_dev, 1),
            "velocity_frequency_signal": round(s_vel, 1),
            "temporal_risk_signal": round(s_time, 1),
            "sequence_pattern_signal": round(s_seq, 1),
        }

        return RiskEvaluationResult(
            transaction_id=str(transaction_id),
            account_id=str(account_id) if account_id else None,
            amount=round(float(amount), 2),
            overall_risk_score=overall_score,
            risk_category=category,
            anomaly_score=round(s_ml, 1),
            contributing_signals=contributing,
            reasons=reasons,
            recommended_action=recommended_action,
        )
