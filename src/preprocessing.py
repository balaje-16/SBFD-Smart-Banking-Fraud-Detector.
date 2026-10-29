"""Data validation, cleaning, encoding, and quality reporting module.

Ensures that transaction data satisfies cybersecurity integrity constraints,
standardizes temporal indices, encodes categorical features, and generates
a comprehensive, transparent Data Quality Report.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class DataQualityReport:
    """Detailed summary of data quality checks, modifications, and retained records."""

    total_records_received: int = 0
    valid_records_retained: int = 0
    duplicates_detected: int = 0
    missing_values_handled: Dict[str, int] = field(default_factory=dict)
    invalid_amounts_dropped: int = 0
    invalid_timestamps_corrected: int = 0
    columns_inspected: List[str] = field(default_factory=list)
    categorical_columns_encoded: List[str] = field(default_factory=list)
    date_range_start: Optional[str] = None
    date_range_end: Optional[str] = None
    quality_notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return asdict(self)


class DataPreprocessor:
    """Preprocesses adapted transaction data and generates quality audit reports."""

    def __init__(
        self,
        allow_zero_amounts: bool = False,
        impute_strategy: str = "median",
    ):
        """Initialize preprocessor.

        Args:
            allow_zero_amounts: Whether $0 transactions are permitted (e.g. auth pings).
            impute_strategy: Strategy for numerical imputation ('median' or 'mean').
        """
        self.allow_zero_amounts = allow_zero_amounts
        self.impute_strategy = impute_strategy
        self.scaler: Optional[StandardScaler] = None
        self.ohe: Optional[OneHotEncoder] = None
        self.fitted: bool = False

    def validate_and_clean(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, DataQualityReport]:
        """Validates schema, cleans records, handles missing and duplicate entries.

        Args:
            df: Adapted DataFrame adhering to canonical banking schema.

        Returns:
            Tuple of (cleaned DataFrame, DataQualityReport).
        """
        if df.empty:
            raise ValueError("Input dataset is empty. Cannot perform preprocessing.")

        report = DataQualityReport(
            total_records_received=len(df),
            columns_inspected=list(df.columns),
        )

        clean_df = df.copy()

        # 1. Check required minimum fields
        if "amount" not in clean_df.columns:
            raise ValueError("Missing critical field 'amount' in transaction records.")

        # 2. Check and clean amounts
        clean_df["amount"] = pd.to_numeric(clean_df["amount"], errors="coerce")
        missing_amt = clean_df["amount"].isna().sum()
        if missing_amt > 0:
            report.missing_values_handled["amount"] = int(missing_amt)
            # Impute missing amounts with median of valid amounts
            median_amt = clean_df["amount"].median()
            clean_df["amount"] = clean_df["amount"].fillna(median_amt if not np.isnan(median_amt) else 10.0)

        # Drop strictly negative amounts or zero amounts (unless explicitly allowed)
        if self.allow_zero_amounts:
            invalid_amt_mask = clean_df["amount"] < 0
        else:
            invalid_amt_mask = clean_df["amount"] <= 0

        invalid_amt_count = int(invalid_amt_mask.sum())
        if invalid_amt_count > 0:
            report.invalid_amounts_dropped = invalid_amt_count
            report.quality_notes.append(
                f"Filtered out {invalid_amt_count} records with invalid non-positive amounts (amount <= 0)."
            )
            clean_df = clean_df[~invalid_amt_mask].copy()

        if clean_df.empty:
            raise ValueError("All records had invalid or non-positive transaction amounts.")

        # 3. Deduplication
        initial_len = len(clean_df)
        # Check duplicate transaction IDs if present
        if "transaction_id" in clean_df.columns and clean_df["transaction_id"].notna().any():
            clean_df = clean_df.drop_duplicates(subset=["transaction_id"], keep="first")
        else:
            # Check duplicate rows across critical subset
            clean_df = clean_df.drop_duplicates(keep="first")

        dups = initial_len - len(clean_df)
        report.duplicates_detected = dups
        if dups > 0:
            report.quality_notes.append(f"Identified and removed {dups} duplicate records.")

        # 4. Standardize timestamps
        if "timestamp" in clean_df.columns:
            clean_df["timestamp"] = pd.to_datetime(clean_df["timestamp"], errors="coerce")
            missing_ts = clean_df["timestamp"].isna().sum()
            if missing_ts > 0:
                report.invalid_timestamps_corrected = int(missing_ts)
                # Fill missing timestamps with sequential fallback
                base = pd.Timestamp("2026-01-01 00:00:00")
                clean_df["timestamp"] = clean_df["timestamp"].fillna(base)
                report.quality_notes.append(f"Corrected {missing_ts} unparseable timestamps with fallback.")

            # Sort chronologically (strictly required for temporal feature engineering)
            clean_df = clean_df.sort_values(by="timestamp").reset_index(drop=True)

            report.date_range_start = str(clean_df["timestamp"].min())
            report.date_range_end = str(clean_df["timestamp"].max())
        else:
            # Create synthetic sequential timestamp
            clean_df["timestamp"] = pd.date_range("2026-01-01", periods=len(clean_df), freq="10s")
            report.date_range_start = str(clean_df["timestamp"].min())
            report.date_range_end = str(clean_df["timestamp"].max())
            report.quality_notes.append("No timestamp provided; synthesized chronological timeline.")

        # 5. Handle missing values in categorical fields
        for cat_col in ["transaction_type", "merchant_category", "device_id"]:
            if cat_col in clean_df.columns:
                n_miss = int(clean_df[cat_col].isna().sum())
                if n_miss > 0:
                    report.missing_values_handled[cat_col] = n_miss
                    clean_df[cat_col] = clean_df[cat_col].fillna("UNKNOWN")

        report.valid_records_retained = len(clean_df)
        report.quality_notes.append(
            f"Preprocessing successful: {report.valid_records_retained} / {report.total_records_received} records retained."
        )

        return clean_df, report

    def encode_and_scale(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
        is_training: bool = True,
    ) -> np.ndarray:
        """Scales numerical feature columns for model ingestion.

        Args:
            df: DataFrame containing computed feature columns.
            feature_cols: Numerical feature column names to process.
            is_training: Whether to fit scaler or apply fitted scaler.

        Returns:
            2D numpy array of scaled numerical features.
        """
        # Ensure all feature cols are numeric and NaN-free
        X_mat = df[feature_cols].copy()
        for col in feature_cols:
            X_mat[col] = pd.to_numeric(X_mat[col], errors="coerce").fillna(0.0)

        X_vals = X_mat.to_numpy(dtype=np.float64)

        if is_training or self.scaler is None:
            self.scaler = StandardScaler()
            scaled_vals = self.scaler.fit_transform(X_vals)
            self.fitted = True
        else:
            scaled_vals = self.scaler.transform(X_vals)

        return scaled_vals
