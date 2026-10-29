"""Account-level behavioural analysis and spending profiling.

Computes longitudinal account statistics, historical baselines,
and velocity metrics to distinguish legitimately large corporate spending
from sudden malicious account takeovers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd


@dataclass
class AccountProfile:
    """Statistical behavioural profile for an individual banking account."""

    account_id: str
    transaction_count: int
    mean_amount: float
    median_amount: float
    std_amount: float
    max_amount: float
    min_amount: float
    first_seen: str
    last_seen: str
    avg_interval_hours: float
    has_sufficient_history: bool
    preferred_types: List[str]
    velocity_95th: float


class BehaviouralAnalyzer:
    """Profiles account-level spending behaviour and detects deviations from baselines."""

    def __init__(self, min_history_records: int = 3):
        """Initialize analyzer.

        Args:
            min_history_records: Minimum transaction count required for a reliable baseline.
        """
        self.min_history_records = min_history_records
        self.profiles: Dict[str, AccountProfile] = {}

    def build_profiles(self, df: pd.DataFrame) -> Dict[str, AccountProfile]:
        """Builds behavioural profiles for all accounts present in the dataset.

        Args:
            df: DataFrame containing at least 'account_id', 'amount', 'timestamp'.

        Returns:
            Dictionary mapping account_id to its AccountProfile.
        """
        if "account_id" not in df.columns or df["account_id"].isna().all():
            self.profiles = {}
            return self.profiles

        grouped = df.groupby("account_id")
        profiles: Dict[str, AccountProfile] = {}

        for acc_id, group in grouped:
            acc_str = str(acc_id)
            n_tx = len(group)
            amounts = group["amount"].astype(float).values
            timestamps = pd.to_datetime(group["timestamp"]).sort_values()

            sufficient = n_tx >= self.min_history_records

            # Calculate intervals if > 1 transaction
            if n_tx > 1:
                diffs = (timestamps.diff().dropna().dt.total_seconds() / 3600.0).values
                avg_interval = float(np.mean(diffs)) if len(diffs) > 0 else 24.0
            else:
                avg_interval = 24.0

            # Velocities (amount / interval)
            velocities = amounts[1:] / (np.maximum(1.0, np.diff(timestamps.astype(np.int64) / 1e9)) / 3600.0) if n_tx > 1 else np.array([amounts[0] / 24.0])
            vel_95th = float(np.percentile(velocities, 95)) if len(velocities) > 0 else float(amounts[0] / 24.0)

            # Preferred transaction types
            if "transaction_type" in group.columns:
                top_types = group["transaction_type"].value_counts().head(3).index.tolist()
            else:
                top_types = ["STANDARD"]

            profiles[acc_str] = AccountProfile(
                account_id=acc_str,
                transaction_count=n_tx,
                mean_amount=round(float(np.mean(amounts)), 2),
                median_amount=round(float(np.median(amounts)), 2),
                std_amount=round(float(np.std(amounts)), 2),
                max_amount=round(float(np.max(amounts)), 2),
                min_amount=round(float(np.min(amounts)), 2),
                first_seen=str(timestamps.min()),
                last_seen=str(timestamps.max()),
                avg_interval_hours=round(avg_interval, 2),
                has_sufficient_history=sufficient,
                preferred_types=top_types,
                velocity_95th=round(vel_95th, 2),
            )

        self.profiles = profiles
        return self.profiles

    def evaluate_transaction_deviation(
        self,
        account_id: Optional[str],
        amount: float,
        timestamp: Optional[pd.Timestamp] = None,
    ) -> Dict[str, Any]:
        """Evaluates whether a transaction is an abnormal deviation for its specific account.

        Args:
            account_id: Account identifier.
            amount: Transaction amount.
            timestamp: Optional transaction timestamp.

        Returns:
            Dictionary containing deviation ratios, z-scores, and contextual remarks.
        """
        if not account_id or str(account_id) not in self.profiles:
            return {
                "has_profile": False,
                "reason": "Account has no historical profile or account_id is absent.",
                "amount_ratio": 1.0,
                "amount_zscore": 0.0,
                "is_outlier_for_account": False,
                "deviation_severity": 0.0,
            }

        profile = self.profiles[str(account_id)]
        if not profile.has_sufficient_history:
            return {
                "has_profile": True,
                "reason": f"Account has only {profile.transaction_count} transaction(s) (insufficient baseline).",
                "amount_ratio": round(amount / max(1.0, profile.mean_amount), 2),
                "amount_zscore": 0.0,
                "is_outlier_for_account": False,
                "deviation_severity": 0.0,
            }

        # Measure against account's personal mean and std
        ratio = amount / max(1.0, profile.mean_amount)
        zscore = (amount - profile.mean_amount) / max(1.0, profile.std_amount)

        # Deviation severity score (0 to 100)
        # Ratio of 1x = 0 severity, ratio of 3x = 40 severity, ratio of 8x+ = 90+ severity
        if ratio <= 1.5:
            severity = 0.0
        elif ratio <= 3.0:
            severity = min(40.0, (ratio - 1.5) * 26.6)
        elif ratio <= 6.0:
            severity = 40.0 + (ratio - 3.0) * 15.0
        else:
            severity = min(100.0, 85.0 + (ratio - 6.0) * 3.0)

        is_outlier = (ratio >= 3.5 and zscore >= 3.0)

        return {
            "has_profile": True,
            "account_id": profile.account_id,
            "account_mean": profile.mean_amount,
            "account_std": profile.std_amount,
            "amount_ratio": round(ratio, 2),
            "amount_zscore": round(zscore, 2),
            "is_outlier_for_account": is_outlier,
            "deviation_severity": round(severity, 1),
            "reason": (
                f"Amount is {ratio:.1f}x the account average (${profile.mean_amount:.2f})"
                if is_outlier
                else "Amount aligns with account spending history."
            ),
        }
