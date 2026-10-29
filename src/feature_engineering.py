"""Feature engineering pipeline for banking transaction fraud detection.

Strictly enforces temporal causality (no lookahead / future leakage) and
provides robust fallbacks for cold-start accounts with minimal history.
Preserves PCA features (e.g. V1..V28) when present without inventing fake metadata.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class FeatureEngineer:
    """Computes behavioural, temporal, and velocity features for banking transactions."""

    def __init__(
        self,
        min_history_records: int = 3,
        short_window_hours: float = 1.0,
        long_window_hours: float = 24.0,
    ):
        """Initialize FeatureEngineer.

        Args:
            min_history_records: Minimum past transactions required for personalized baseline.
            short_window_hours: Short frequency time window in hours.
            long_window_hours: Long frequency time window in hours.
        """
        self.min_history_records = min_history_records
        self.short_window_hours = short_window_hours
        self.long_window_hours = long_window_hours
        self.global_amount_mean: float = 100.0
        self.global_amount_std: float = 50.0

    def compute_features(
        self,
        df: pd.DataFrame,
        is_training: bool = True,
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Compute all causal features for the dataset.

        Args:
            df: Cleaned DataFrame sorted chronologically by timestamp.
            is_training: If True, computes and stores global baselines.

        Returns:
            Tuple of (DataFrame with appended feature columns, list of engineered feature names).
        """
        if df.empty:
            raise ValueError("Empty DataFrame passed to feature engineering.")

        # Ensure chronological ordering
        feat_df = df.sort_values(by="timestamp").copy().reset_index(drop=True)

        # 1. Base Amount & Log Transformation
        feat_df["feature_amount"] = feat_df["amount"].astype(float)
        feat_df["feature_log_amount"] = np.log1p(np.maximum(0.0, feat_df["feature_amount"]))

        if is_training:
            self.global_amount_mean = float(feat_df["feature_amount"].mean())
            self.global_amount_std = float(max(1.0, feat_df["feature_amount"].std()))

        # 2. Temporal Features
        timestamps = pd.to_datetime(feat_df["timestamp"])
        feat_df["feature_hour"] = timestamps.dt.hour
        feat_df["feature_dayofweek"] = timestamps.dt.dayofweek
        feat_df["feature_is_weekend"] = feat_df["feature_dayofweek"].isin([5, 6]).astype(float)
        # Night transactions (23:00 to 05:00) often present heightened risk
        feat_df["feature_is_night"] = feat_df["feature_hour"].apply(
            lambda h: 1.0 if (h >= 23 or h <= 5) else 0.0
        )

        has_accounts = "account_id" in feat_df.columns and feat_df["account_id"].notna().any()

        # 3. Account-Level Behavioural Features (Causal past-only computation)
        if has_accounts:
            feat_df = self._compute_account_features(feat_df)
        else:
            feat_df = self._compute_global_sequence_features(feat_df)

        # 4. Categorical Encoding (Transaction Types)
        cat_features: List[str] = []
        if "transaction_type" in feat_df.columns:
            types = feat_df["transaction_type"].fillna("UNKNOWN").astype(str).str.upper()
            # Common banking types
            for t_name in ["TRANSFER", "WIRE", "PAYMENT", "CASH_OUT", "ATM", "ONLINE"]:
                col_name = f"feature_type_{t_name.lower()}"
                feat_df[col_name] = types.apply(lambda x: 1.0 if t_name in x else 0.0)
                cat_features.append(col_name)

        # 5. Anonymized / Preserved Features (e.g. V1..V28 for Kaggle CC)
        preserved_cols = [c for c in feat_df.columns if c.lower().startswith("v") and c[1:].isdigit()]
        for p_col in preserved_cols:
            feat_df[p_col] = pd.to_numeric(feat_df[p_col], errors="coerce").fillna(0.0)

        # Compile final feature list for ML model ingestion
        base_features = [
            "feature_amount",
            "feature_log_amount",
            "feature_hour",
            "feature_is_weekend",
            "feature_is_night",
            "feature_delta_time_sec",
            "feature_amount_ratio_hist",
            "feature_amount_zscore_hist",
            "feature_tx_velocity",
            "feature_freq_1h",
            "feature_freq_24h",
            "feature_is_cold_start",
        ]
        all_features = base_features + cat_features + preserved_cols

        return feat_df, all_features

    def _compute_account_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Causal rolling calculations per account using only past transactions."""
        n = len(df)
        delta_time_sec = np.zeros(n, dtype=np.float64)
        amount_ratio = np.ones(n, dtype=np.float64)
        amount_zscore = np.zeros(n, dtype=np.float64)
        tx_velocity = np.zeros(n, dtype=np.float64)
        freq_1h = np.zeros(n, dtype=np.float64)
        freq_24h = np.zeros(n, dtype=np.float64)
        is_cold_start = np.zeros(n, dtype=np.float64)

        # Group by account and iterate sequentially to ensure strict temporal causality
        # Tracking dictionaries per account
        account_history: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}

        timestamps = pd.to_datetime(df["timestamp"]).tolist()
        amounts = df["feature_amount"].tolist()
        account_ids = df["account_id"].tolist()

        one_hour = pd.Timedelta(hours=self.short_window_hours)
        one_day = pd.Timedelta(hours=self.long_window_hours)

        for i in range(n):
            acc = str(account_ids[i]) if pd.notna(account_ids[i]) else "UNKNOWN"
            current_time = timestamps[i]
            current_amt = amounts[i]

            hist = account_history.get(acc, [])

            if len(hist) < self.min_history_records:
                # Cold-start fallback: Account lacks sufficient history
                is_cold_start[i] = 1.0
                delta_time_sec[i] = 3600.0  # neutral 1 hour default
                amount_ratio[i] = current_amt / (self.global_amount_mean + 1e-5)
                amount_zscore[i] = (current_amt - self.global_amount_mean) / self.global_amount_std
                tx_velocity[i] = current_amt / (delta_time_sec[i] + 60.0)
                freq_1h[i] = 1.0
                freq_24h[i] = 1.0
            else:
                is_cold_start[i] = 0.0
                last_time, _ = hist[-1]
                delta_s = max(1.0, (current_time - last_time).total_seconds())
                delta_time_sec[i] = delta_s

                # Historical past amounts
                past_amounts = [h[1] for h in hist]
                hist_mean = float(np.mean(past_amounts))
                hist_std = float(max(1.0, np.std(past_amounts)))

                amount_ratio[i] = current_amt / (hist_mean + 1e-5)
                amount_zscore[i] = (current_amt - hist_mean) / hist_std
                tx_velocity[i] = current_amt / (delta_s + 60.0)

                # Causal window counts (strictly prior within window)
                f1 = sum(1 for (t, _) in hist if (current_time - t) <= one_hour)
                f24 = sum(1 for (t, _) in hist if (current_time - t) <= one_day)
                freq_1h[i] = float(f1 + 1)
                freq_24h[i] = float(f24 + 1)

            # Append current transaction to account history for subsequent events
            if acc not in account_history:
                account_history[acc] = []
            account_history[acc].append((current_time, current_amt))

        df["feature_delta_time_sec"] = delta_time_sec
        df["feature_amount_ratio_hist"] = np.clip(amount_ratio, 0.0, 100.0)
        df["feature_amount_zscore_hist"] = np.clip(amount_zscore, -5.0, 30.0)
        df["feature_tx_velocity"] = np.clip(tx_velocity, 0.0, 1000.0)
        df["feature_freq_1h"] = freq_1h
        df["feature_freq_24h"] = freq_24h
        df["feature_is_cold_start"] = is_cold_start

        return df

    def _compute_global_sequence_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fallback features for datasets lacking account IDs (e.g. Kaggle Credit Card)."""
        n = len(df)
        timestamps = pd.to_datetime(df["timestamp"]).tolist()
        amounts = df["feature_amount"].tolist()

        delta_time_sec = np.zeros(n, dtype=np.float64)
        amount_ratio = np.ones(n, dtype=np.float64)
        amount_zscore = np.zeros(n, dtype=np.float64)
        tx_velocity = np.zeros(n, dtype=np.float64)
        freq_1h = np.ones(n, dtype=np.float64)
        freq_24h = np.ones(n, dtype=np.float64)

        one_hour = pd.Timedelta(hours=1.0)
        recent_window: List[Tuple[pd.Timestamp, float]] = []

        for i in range(n):
            current_time = timestamps[i]
            current_amt = amounts[i]

            if i > 0:
                delta_s = max(1.0, (current_time - timestamps[i - 1]).total_seconds())
            else:
                delta_s = 60.0
            delta_time_sec[i] = delta_s

            amount_ratio[i] = current_amt / (self.global_amount_mean + 1e-5)
            amount_zscore[i] = (current_amt - self.global_amount_mean) / self.global_amount_std
            tx_velocity[i] = current_amt / (delta_s + 60.0)

            # Stream burst count
            recent_window = [(t, a) for (t, a) in recent_window if (current_time - t) <= one_hour]
            freq_1h[i] = float(len(recent_window) + 1)
            freq_24h[i] = float(len(recent_window) + 1)
            recent_window.append((current_time, current_amt))

        df["feature_delta_time_sec"] = delta_time_sec
        df["feature_amount_ratio_hist"] = np.clip(amount_ratio, 0.0, 100.0)
        df["feature_amount_zscore_hist"] = np.clip(amount_zscore, -5.0, 30.0)
        df["feature_tx_velocity"] = np.clip(tx_velocity, 0.0, 1000.0)
        df["feature_freq_1h"] = freq_1h
        df["feature_freq_24h"] = freq_24h
        df["feature_is_cold_start"] = 1.0  # Flagged because individualized history is absent

        return df
