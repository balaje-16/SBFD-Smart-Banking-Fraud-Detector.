"""Rule-based suspicious transaction sequence and cyber-heist kill chain detection.

Detects temporal transaction bursts, rapid account draining, and correlates
synthetic authentication anomalies with subsequent fraudulent transfers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import timedelta
from typing import Any, Dict, List, Optional
import pandas as pd


@dataclass
class SequenceViolation:
    """Represents a flagged multi-event suspicious sequence."""

    rule_name: str
    account_id: str
    trigger_transaction_id: str
    risk_contribution: float
    description: str
    involved_transaction_ids: List[str] = field(default_factory=list)
    timestamp: Optional[str] = None


class SequenceDetector:
    """Rule-based engine detecting high-velocity and burst transaction sequences."""

    def __init__(
        self,
        burst_time_window_minutes: int = 15,
        burst_count_threshold: int = 3,
        drain_transfer_window_minutes: int = 30,
        drain_count_threshold: int = 2,
        drain_amount_multiplier: float = 3.0,
        dormancy_days_threshold: int = 14,
    ):
        """Initialize SequenceDetector.

        Args:
            burst_time_window_minutes: Maximum window to consider a burst.
            burst_count_threshold: Minimum transactions within window to trigger burst alarm.
            drain_transfer_window_minutes: Window for rapid high-value drain attempts.
            drain_count_threshold: Minimum large transfers within window.
            drain_amount_multiplier: Multiplier of historical average considered draining.
            dormancy_days_threshold: Days of silence before reactivation alert.
        """
        self.burst_window = timedelta(minutes=burst_time_window_minutes)
        self.burst_count = burst_count_threshold
        self.drain_window = timedelta(minutes=drain_transfer_window_minutes)
        self.drain_count = drain_count_threshold
        self.drain_mult = drain_amount_multiplier
        self.dormancy_days = timedelta(days=dormancy_days_threshold)

    def detect_transaction_sequences(
        self,
        df: pd.DataFrame,
        account_baselines: Optional[Dict[str, float]] = None,
    ) -> List[SequenceViolation]:
        """Scans chronological transactions for suspicious temporal sequences.

        Args:
            df: DataFrame sorted chronologically with at least 'timestamp', 'amount', 'transaction_id'.
            account_baselines: Optional map of account_id -> historical average amount.

        Returns:
            List of detected SequenceViolation objects.
        """
        violations: List[SequenceViolation] = []
        if df.empty or "account_id" not in df.columns or df["account_id"].isna().all():
            return violations

        baselines = account_baselines or {}
        df_sorted = df.sort_values(by="timestamp").copy()

        # Group by account to inspect chronological sequence per account
        for acc_id, group in df_sorted.groupby("account_id"):
            acc_str = str(acc_id)
            records = group.to_dict("records")
            n = len(records)
            if n < 2:
                continue

            base_mean = baselines.get(acc_str, group["amount"].median())

            # 1. Sliding window check for rapid bursts
            for i in range(n):
                current = records[i]
                c_time = pd.to_datetime(current["timestamp"])

                # Look back within burst window
                burst_txs = [
                    records[j]
                    for j in range(max(0, i - 15), i + 1)
                    if (c_time - pd.to_datetime(records[j]["timestamp"])) <= self.burst_window
                ]

                if len(burst_txs) >= self.burst_count:
                    violations.append(
                        SequenceViolation(
                            rule_name="RAPID_TRANSACTION_BURST",
                            account_id=acc_str,
                            trigger_transaction_id=str(current.get("transaction_id", f"TX_{i}")),
                            risk_contribution=75.0,
                            description=(
                                f"High-velocity burst: {len(burst_txs)} transactions executed within "
                                f"{self.burst_window.total_seconds() / 60:.0f} minutes."
                            ),
                            involved_transaction_ids=[str(t.get("transaction_id", "")) for t in burst_txs],
                            timestamp=str(c_time),
                        )
                    )

                # 2. Rapid high-value draining transfers
                drain_txs = [
                    records[j]
                    for j in range(max(0, i - 10), i + 1)
                    if (c_time - pd.to_datetime(records[j]["timestamp"])) <= self.drain_window
                    and float(records[j]["amount"]) >= (base_mean * self.drain_mult)
                ]

                if len(drain_txs) >= self.drain_count:
                    violations.append(
                        SequenceViolation(
                            rule_name="RAPID_ACCOUNT_DRAIN",
                            account_id=acc_str,
                            trigger_transaction_id=str(current.get("transaction_id", f"TX_{i}")),
                            risk_contribution=90.0,
                            description=(
                                f"Account draining pattern: {len(drain_txs)} high-value transfers "
                                f"(each > {self.drain_mult:.1f}x average) in under {self.drain_window.total_seconds() / 60:.0f} mins."
                            ),
                            involved_transaction_ids=[str(t.get("transaction_id", "")) for t in drain_txs],
                            timestamp=str(c_time),
                        )
                    )

                # 3. Dormancy reactivation followed by sudden large transfer
                if i > 0:
                    prev = records[i - 1]
                    p_time = pd.to_datetime(prev["timestamp"])
                    gap = c_time - p_time
                    if gap >= self.dormancy_days and float(current["amount"]) >= (base_mean * 2.0):
                        violations.append(
                            SequenceViolation(
                                rule_name="DORMANT_ACCOUNT_SUDDEN_SPIKE",
                                account_id=acc_str,
                                trigger_transaction_id=str(current.get("transaction_id", f"TX_{i}")),
                                risk_contribution=65.0,
                                description=(
                                    f"Sudden high-value transfer (${float(current['amount']):.2f}) after "
                                    f"{gap.days} days of account inactivity."
                                ),
                                involved_transaction_ids=[
                                    str(prev.get("transaction_id", "")),
                                    str(current.get("transaction_id", "")),
                                ],
                                timestamp=str(c_time),
                            )
                        )

        return violations

    @staticmethod
    def correlate_auth_events(
        df_transactions: pd.DataFrame,
        df_auth_events: pd.DataFrame,
        window_minutes: int = 60,
    ) -> List[Dict[str, Any]]:
        """Correlates simulated authentication events (brute force, device change) with transactions.

        Important: Only applicable when synthetic authentication event logs are provided.
        Transaction-only datasets do not support this correlation.

        Returns:
            List of correlated cyber-heist kill chain alerts.
        """
        correlated_alerts: List[Dict[str, Any]] = []
        if df_auth_events.empty or df_transactions.empty:
            return correlated_alerts

        auth_df = df_auth_events.copy()
        auth_df["timestamp"] = pd.to_datetime(auth_df["timestamp"])
        tx_df = df_transactions.copy()
        tx_df["timestamp"] = pd.to_datetime(tx_df["timestamp"])

        # Find accounts with failed logins or new device logins
        suspicious_auth = auth_df[
            auth_df["event_type"].isin(["LOGIN_FAILED", "SECURITY_PHONE_UPDATED", "BENEFICIARY_ADDED"])
        ]

        window = timedelta(minutes=window_minutes)

        for _, auth_row in suspicious_auth.iterrows():
            acc = auth_row["account_id"]
            auth_time = auth_row["timestamp"]

            # Check if matching transactions occurred shortly thereafter
            subsequent_txs = tx_df[
                (tx_df["account_id"] == acc)
                & (tx_df["timestamp"] >= auth_time)
                & (tx_df["timestamp"] <= auth_time + window)
            ]

            for _, tx in subsequent_txs.iterrows():
                correlated_alerts.append({
                    "cyber_kill_chain_stage": "Account Takeover & Subsequent Transfer",
                    "account_id": acc,
                    "transaction_id": tx.get("transaction_id"),
                    "amount": tx.get("amount"),
                    "auth_event_type": auth_row.get("event_type"),
                    "auth_event_time": str(auth_time),
                    "transaction_time": str(tx["timestamp"]),
                    "delay_minutes": round((tx["timestamp"] - auth_time).total_seconds() / 60.0, 1),
                    "risk_boost": 40.0,
                    "evidence": (
                        f"Transaction occurred {round((tx['timestamp'] - auth_time).total_seconds() / 60.0, 1)}m "
                        f"after suspicious authentication event '{auth_row.get('event_type')}' from IP {auth_row.get('ip_address')}."
                    ),
                })

        return correlated_alerts
