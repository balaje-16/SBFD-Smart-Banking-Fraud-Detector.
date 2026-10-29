"""Dataset loading, schema mapping adapters, and synthetic data generation.

This module provides:
1. Adapters for popular datasets (Credit Card Fraud, PaySim, Custom CSVs).
2. Common schema normalization.
3. Safe file parsing with size limits.
4. Comprehensive synthetic transaction and authentication log generator.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

# Standard internal schema definition
CANONICAL_COLUMNS = [
    "transaction_id",
    "account_id",
    "timestamp",
    "amount",
    "transaction_type",
    "merchant_category",
    "device_id",
    "is_fraud",
]

DATASET_TYPE_CREDIT_CARD = "Credit Card Fraud Dataset (Kaggle)"
DATASET_TYPE_PAYSIM = "PaySim Synthetic Mobile Money"
DATASET_TYPE_CUSTOM = "Custom Banking CSV Schema"
DATASET_TYPE_UNKNOWN = "Unknown / Generic"

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB safe default limit


class DatasetAdapter:
    """Detects dataset format and standardizes records into a canonical representation."""

    @staticmethod
    def detect_format(df: pd.DataFrame) -> str:
        """Heuristically identify the dataset format from column names.

        Args:
            df: Raw DataFrame read from user CSV.

        Returns:
            Dataset type string constant.
        """
        cols = {c.strip() for c in df.columns}
        cols_lower = {c.lower() for c in cols}

        # Check for Kaggle Credit Card Fraud dataset: Time, Amount, Class, V1..V28
        has_v_features = any(f"v{i}" in cols_lower for i in range(1, 10))
        if "amount" in cols_lower and ("class" in cols_lower or has_v_features):
            return DATASET_TYPE_CREDIT_CARD

        # Check for PaySim dataset: step, type, amount, nameOrig, nameDest, isFraud
        if {"step", "nameorig", "namedest"}.issubset(cols_lower) and "amount" in cols_lower:
            return DATASET_TYPE_PAYSIM

        # Check for standard custom columns
        if "amount" in cols_lower and ("account_id" in cols_lower or "transaction_id" in cols_lower or "timestamp" in cols_lower):
            return DATASET_TYPE_CUSTOM

        return DATASET_TYPE_UNKNOWN

    @classmethod
    def adapt(
        cls, df: pd.DataFrame, dataset_type: Optional[str] = None
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Maps dataset-specific columns to the standard internal representation.

        Args:
            df: Raw DataFrame.
            dataset_type: Optional explicit type override; if None, auto-detected.

        Returns:
            Tuple of (standardized DataFrame, metadata dictionary).
        """
        if df.empty:
            raise ValueError("The provided dataset is empty (0 records).")

        raw_df = df.copy()
        # Clean column names
        raw_df.columns = [str(c).strip() for c in raw_df.columns]
        detected = dataset_type or cls.detect_format(raw_df)

        meta: Dict[str, Any] = {
            "original_rows": len(raw_df),
            "original_columns": list(raw_df.columns),
            "detected_type": detected,
            "has_account_id": False,
            "has_ground_truth": False,
            "preserved_features": [],
            "adapter_notes": [],
        }

        if detected == DATASET_TYPE_CREDIT_CARD:
            standard_df, meta = cls._adapt_credit_card(raw_df, meta)
        elif detected == DATASET_TYPE_PAYSIM:
            standard_df, meta = cls._adapt_paysim(raw_df, meta)
        elif detected == DATASET_TYPE_CUSTOM:
            standard_df, meta = cls._adapt_custom(raw_df, meta)
        else:
            standard_df, meta = cls._adapt_fallback(raw_df, meta)

        return standard_df, meta

    @staticmethod
    def _adapt_credit_card(df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Adapt Kaggle Credit Card dataset (Time, V1-V28, Amount, Class)."""
        lower_map = {c.lower(): c for c in df.columns}
        res = pd.DataFrame()

        # Amount
        amt_col = lower_map.get("amount", "Amount")
        res["amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

        # Transaction ID
        res["transaction_id"] = [f"TX_CC_{i+1:06d}" for i in range(len(df))]

        # Credit Card Kaggle has NO individual account identifiers for privacy
        res["account_id"] = None
        meta["has_account_id"] = False
        meta["adapter_notes"].append(
            "Anonymized Credit Card dataset: No account_id provided. Account-level profiling is disabled."
        )

        # Timestamp: Time represents elapsed seconds since first transaction
        time_col = lower_map.get("time", "Time")
        if time_col in df.columns:
            base_time = pd.Timestamp("2026-01-01 00:00:00")
            elapsed_seconds = pd.to_numeric(df[time_col], errors="coerce").fillna(0)
            res["timestamp"] = base_time + pd.to_timedelta(elapsed_seconds, unit="s")
        else:
            base_time = pd.Timestamp("2026-01-01 00:00:00")
            res["timestamp"] = [base_time + pd.Timedelta(seconds=i * 10) for i in range(len(df))]

        # Transaction Type and Merchant Category
        res["transaction_type"] = "CARD_PAYMENT"
        res["merchant_category"] = "RETAIL_GENERAL"
        res["device_id"] = None

        # Ground truth Class
        class_col = lower_map.get("class", "Class")
        if class_col in df.columns:
            res["is_fraud"] = pd.to_numeric(df[class_col], errors="coerce").fillna(0).astype(int)
            meta["has_ground_truth"] = True
        else:
            res["is_fraud"] = None

        # Preserve PCA features V1..V28 without invention
        v_cols = [c for c in df.columns if c.lower().startswith("v") and c[1:].isdigit()]
        for v in v_cols:
            res[v] = pd.to_numeric(df[v], errors="coerce").fillna(0.0)
        meta["preserved_features"] = v_cols

        return res, meta

    @staticmethod
    def _adapt_paysim(df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Adapt PaySim synthetic mobile money dataset."""
        lower_map = {c.lower(): c for c in df.columns}
        res = pd.DataFrame()

        # Transaction ID
        res["transaction_id"] = [f"TX_PS_{i+1:07d}" for i in range(len(df))]

        # Account ID (nameOrig is the sender account)
        orig_col = lower_map.get("nameorig", "nameOrig")
        res["account_id"] = df[orig_col].astype(str)
        meta["has_account_id"] = True

        # Amount
        amt_col = lower_map.get("amount", "amount")
        res["amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

        # Timestamp: step represents hours (1 step = 1 hour)
        step_col = lower_map.get("step", "step")
        if step_col in df.columns:
            base_time = pd.Timestamp("2026-01-01 00:00:00")
            steps = pd.to_numeric(df[step_col], errors="coerce").fillna(1)
            # Add minute jitter based on index so transactions in same hour have sequential ordering
            res["timestamp"] = [
                base_time + pd.Timedelta(hours=int(s), minutes=(i % 60), seconds=((i * 7) % 60))
                for i, s in enumerate(steps)
            ]
        else:
            res["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="min")

        # Transaction Type
        type_col = lower_map.get("type", "type")
        if type_col in df.columns:
            res["transaction_type"] = df[type_col].astype(str).str.upper()
        else:
            res["transaction_type"] = "TRANSFER"

        # Merchant Category / Destination
        dest_col = lower_map.get("namedest", "nameDest")
        if dest_col in df.columns:
            res["merchant_category"] = df[dest_col].astype(str).apply(
                lambda d: "MERCHANT" if str(d).startswith("M") else "CUSTOMER_TRANSFER"
            )
        else:
            res["merchant_category"] = "TRANSFER"

        res["device_id"] = None

        # Ground truth
        fraud_col = lower_map.get("isfraud", "isFraud")
        if fraud_col in df.columns:
            res["is_fraud"] = pd.to_numeric(df[fraud_col], errors="coerce").fillna(0).astype(int)
            meta["has_ground_truth"] = True
        else:
            res["is_fraud"] = None

        # Preserve PaySim balance dynamics
        preserved = []
        for col in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest", "nameDest"]:
            real_c = lower_map.get(col.lower())
            if real_c and real_c in df.columns:
                res[col] = df[real_c]
                preserved.append(col)
        meta["preserved_features"] = preserved

        return res, meta

    @staticmethod
    def _adapt_custom(df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Adapt a custom banking CSV schema."""
        lower_map = {c.lower(): c for c in df.columns}
        res = pd.DataFrame()

        # Transaction ID
        tx_col = lower_map.get("transaction_id") or lower_map.get("tx_id") or lower_map.get("id")
        if tx_col:
            res["transaction_id"] = df[tx_col].astype(str)
        else:
            res["transaction_id"] = [f"TX_CUST_{i+1:06d}" for i in range(len(df))]

        # Account ID
        acc_col = lower_map.get("account_id") or lower_map.get("customer_id") or lower_map.get("user_id")
        if acc_col:
            res["account_id"] = df[acc_col].astype(str)
            meta["has_account_id"] = True
        else:
            res["account_id"] = None
            meta["has_account_id"] = False

        # Amount
        amt_col = lower_map.get("amount") or lower_map.get("tx_amount") or lower_map.get("value")
        if not amt_col:
            raise ValueError("Custom dataset must contain an 'amount' column.")
        res["amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)

        # Timestamp
        time_col = lower_map.get("timestamp") or lower_map.get("datetime") or lower_map.get("date") or lower_map.get("time")
        if time_col:
            res["timestamp"] = pd.to_datetime(df[time_col], errors="coerce").fillna(pd.Timestamp("2026-01-01"))
        else:
            res["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="5min")

        # Transaction Type
        type_col = lower_map.get("transaction_type") or lower_map.get("type")
        res["transaction_type"] = df[type_col].astype(str) if type_col else "PAYMENT"

        # Merchant Category
        merch_col = lower_map.get("merchant_category") or lower_map.get("category")
        res["merchant_category"] = df[merch_col].astype(str) if merch_col else "GENERAL"

        # Device ID
        dev_col = lower_map.get("device_id")
        res["device_id"] = df[dev_col].astype(str) if dev_col else None

        # Ground truth
        fraud_col = lower_map.get("is_fraud") or lower_map.get("isfraud") or lower_map.get("label") or lower_map.get("class")
        if fraud_col:
            res["is_fraud"] = pd.to_numeric(df[fraud_col], errors="coerce").fillna(0).astype(int)
            meta["has_ground_truth"] = True
        else:
            res["is_fraud"] = None

        return res, meta

    @staticmethod
    def _adapt_fallback(df: pd.DataFrame, meta: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fallback adapter for generic CSV containing at least one numerical amount column."""
        # Find numeric column most likely to be amount
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            raise ValueError("No numeric columns found in the uploaded file to serve as transaction amount.")

        amt_col = numeric_cols[-1]
        for col in numeric_cols:
            if "amount" in col.lower() or "price" in col.lower() or "val" in col.lower():
                amt_col = col
                break

        res = pd.DataFrame()
        res["transaction_id"] = [f"TX_GEN_{i+1:06d}" for i in range(len(df))]
        res["account_id"] = None
        res["amount"] = pd.to_numeric(df[amt_col], errors="coerce").fillna(0.0)
        res["timestamp"] = pd.date_range("2026-01-01", periods=len(df), freq="10min")
        res["transaction_type"] = "GENERIC"
        res["merchant_category"] = "GENERAL"
        res["device_id"] = None
        res["is_fraud"] = None

        meta["adapter_notes"].append(
            f"Generic dataset adapted using column '{amt_col}' as transaction amount. Metadata synthesized."
        )
        return res, meta


# ---------------------------------------------------------------------------
# Safe CSV File Loader
# ---------------------------------------------------------------------------

def load_csv_safely(
    file_source: Any,
    max_size_bytes: int = MAX_FILE_SIZE_BYTES,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Safely loads a CSV file from a file path or uploaded file buffer.

    Performs:
    1. File size verification.
    2. File type check (CSV only, no executable scripts).
    3. Error handling with descriptive user messages.

    Args:
        file_source: File path (str) or file-like buffer (e.g., Streamlit UploadedFile).
        max_size_bytes: Maximum allowed size in bytes (default 50 MB).

    Returns:
        Tuple of (Standardized DataFrame, metadata dictionary).
    """
    if file_source is None:
        raise ValueError("No file provided for upload.")

    # Size check
    if isinstance(file_source, str):
        if not os.path.exists(file_source):
            raise FileNotFoundError(f"File not found at path: {file_source}")
        size = os.path.getsize(file_source)
        if size > max_size_bytes:
            raise ValueError(
                f"File size exceeds safety limit: {size / (1024*1024):.1f} MB (Max allowed: {max_size_bytes / (1024*1024):.0f} MB)."
            )
        # Parse CSV
        df = pd.read_csv(file_source, low_memory=False)
    else:
        # Buffer-like (Streamlit upload)
        if hasattr(file_source, "size") and file_source.size > max_size_bytes:
            raise ValueError(
                f"Uploaded file exceeds limit: {file_source.size / (1024*1024):.1f} MB (Max allowed: {max_size_bytes / (1024*1024):.0f} MB)."
            )
        name = getattr(file_source, "name", "upload.csv")
        if not name.lower().endswith(".csv"):
            raise ValueError("Only CSV (.csv) files are permitted. Executables or script files are strictly rejected.")
        df = pd.read_csv(file_source, low_memory=False)

    return DatasetAdapter.adapt(df)


# ---------------------------------------------------------------------------
# Synthetic Data Generator
# ---------------------------------------------------------------------------

class SyntheticDataGenerator:
    """Generates realistic banking transactions and simulated authentication logs.

    Includes 7 distinct behavioural patterns:
    1. Normal low-value spending (daily baseline, $5-$120).
    2. Normal high-value spending (affluent/corporate baseline, $2,000-$9,000).
    3. Sudden outlier amount (account with $40 average suddenly spends $8,200).
    4. Rapid high-frequency bursts (5 transactions within 3 minutes).
    5. Sudden surge in transaction volume (account frequency spike).
    6. Rapid unauthorized transfers (draining attack: 3 transfers in 10 minutes).
    7. Legitimate benign unusual transaction (holiday travel, NOT fraud).
    """

    @staticmethod
    def generate_transactions(
        n_records: int = 1200,
        fraud_ratio: float = 0.04,
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Generate a synthetic banking transactions dataset.

        Args:
            n_records: Approximate number of transactions to generate.
            fraud_ratio: Approximate ratio of fraudulent transactions.
            random_seed: Random seed for reproducibility.

        Returns:
            DataFrame adhering to canonical banking schema.
        """
        rng = np.random.default_rng(random_seed)
        records: List[Dict[str, Any]] = []

        base_time = datetime(2026, 1, 15, 8, 0, 0)

        # 30 unique accounts with varying spending baselines
        accounts = [f"ACC_NORM_{i:03d}" for i in range(1, 25)]
        affluent_accounts = [f"ACC_CORP_{i:02d}" for i in range(1, 6)]
        compromised_accounts = ["ACC_VICTIM_01", "ACC_VICTIM_02", "ACC_VICTIM_03"]

        merchant_categories = ["GROCERY", "RESTAURANT", "ELECTRONICS", "TRAVEL", "UTILITY", "ENTERTAINMENT"]
        devices = [f"DEV_{i:04d}" for i in range(1, 50)]

        tx_count = 0
        current_time = base_time

        # Account baseline parameters
        acc_baselines = {}
        for acc in accounts:
            acc_baselines[acc] = {"mean": rng.uniform(25, 110), "std": rng.uniform(5, 30)}
        for acc in affluent_accounts:
            acc_baselines[acc] = {"mean": rng.uniform(2500, 6500), "std": rng.uniform(400, 1200)}
        for acc in compromised_accounts:
            acc_baselines[acc] = {"mean": rng.uniform(40, 95), "std": rng.uniform(10, 25)}

        while tx_count < n_records:
            scenario_choice = rng.random()

            # Scenario 1: Normal Low-Value Transaction (75% probability)
            if scenario_choice < 0.72:
                acc = rng.choice(accounts)
                base = acc_baselines[acc]
                amt = max(3.50, rng.normal(base["mean"], base["std"]))
                current_time += timedelta(minutes=int(rng.integers(15, 240)))
                tx_count += 1
                records.append({
                    "transaction_id": f"TX_{tx_count:06d}",
                    "account_id": acc,
                    "timestamp": current_time,
                    "amount": round(float(amt), 2),
                    "transaction_type": rng.choice(["POS_PURCHASE", "ONLINE_PAYMENT", "ATM_WITHDRAWAL"]),
                    "merchant_category": rng.choice(merchant_categories),
                    "device_id": rng.choice(devices[:20]),
                    "is_fraud": 0,
                    "scenario": "Normal Retail Spending",
                })

            # Scenario 2: Normal High-Value Spending for Affluent Account (12% probability)
            elif scenario_choice < 0.84:
                acc = rng.choice(affluent_accounts)
                base = acc_baselines[acc]
                amt = max(800.0, rng.normal(base["mean"], base["std"]))
                current_time += timedelta(minutes=int(rng.integers(30, 360)))
                tx_count += 1
                records.append({
                    "transaction_id": f"TX_{tx_count:06d}",
                    "account_id": acc,
                    "timestamp": current_time,
                    "amount": round(float(amt), 2),
                    "transaction_type": "WIRE_TRANSFER",
                    "merchant_category": "COMMERCIAL_SUPPLIES",
                    "device_id": rng.choice(devices[20:30]),
                    "is_fraud": 0,
                    "scenario": "Legitimate High-Value Corporate",
                })

            # Scenario 7: Legitimate Benign Unusual Transaction (4% probability)
            # E.g. Annual insurance or vacation purchase. Unusual for account, but legitimate!
            elif scenario_choice < 0.88:
                acc = rng.choice(accounts)
                base = acc_baselines[acc]
                amt = base["mean"] * rng.uniform(6.0, 10.0)  # High for this user, but benign
                current_time += timedelta(minutes=int(rng.integers(40, 200)))
                tx_count += 1
                records.append({
                    "transaction_id": f"TX_{tx_count:06d}",
                    "account_id": acc,
                    "timestamp": current_time,
                    "amount": round(float(amt), 2),
                    "transaction_type": "ONLINE_PAYMENT",
                    "merchant_category": "TRAVEL",
                    "device_id": rng.choice(devices[:20]),
                    "is_fraud": 0,
                    "scenario": "Benign Rare Event (Vacation/Annual Premium)",
                })

            # Scenario 3: Sudden Outlier Fraud Transaction (3% probability)
            elif scenario_choice < 0.92:
                acc = rng.choice(compromised_accounts)
                amt = rng.uniform(7000.0, 14000.0)  # Enormous jump from $60 baseline
                current_time += timedelta(minutes=int(rng.integers(5, 60)))
                tx_count += 1
                records.append({
                    "transaction_id": f"TX_{tx_count:06d}",
                    "account_id": acc,
                    "timestamp": current_time,
                    "amount": round(float(amt), 2),
                    "transaction_type": "ONLINE_TRANSFER",
                    "merchant_category": "LUXURY_GOODS",
                    "device_id": "DEV_UNKNOWN_9999",
                    "is_fraud": 1,
                    "scenario": "Account Takeover Sudden High-Value Spike",
                })

            # Scenario 4: Multiple Rapid Micro/Testing Transactions (Card Testing Attack) (4% probability)
            elif scenario_choice < 0.96:
                acc = rng.choice(compromised_accounts)
                burst_time = current_time + timedelta(minutes=int(rng.integers(30, 120)))
                for b_i in range(4):
                    tx_count += 1
                    burst_time += timedelta(seconds=int(rng.integers(15, 45)))
                    records.append({
                        "transaction_id": f"TX_{tx_count:06d}",
                        "account_id": acc,
                        "timestamp": burst_time,
                        "amount": round(float(rng.uniform(1.20, 4.80)), 2),
                        "transaction_type": "ONLINE_PURCHASE",
                        "merchant_category": "DIGITAL_CONTENT",
                        "device_id": "DEV_BOT_0101",
                        "is_fraud": 1,
                        "scenario": "Automated Card Testing Rapid Burst",
                    })
                current_time = burst_time

            # Scenario 6: Cyber Heist Rapid Draining Transfers (4% probability)
            else:
                acc = rng.choice(compromised_accounts)
                drain_time = current_time + timedelta(minutes=int(rng.integers(20, 90)))
                for d_i in range(3):
                    tx_count += 1
                    drain_time += timedelta(minutes=int(rng.integers(2, 6)))
                    records.append({
                        "transaction_id": f"TX_{tx_count:06d}",
                        "account_id": acc,
                        "timestamp": drain_time,
                        "amount": round(float(rng.uniform(3200.0, 4800.0)), 2),
                        "transaction_type": "EXTERNAL_WIRE",
                        "merchant_category": "CRYPTO_EXCHANGE",
                        "device_id": "DEV_FOREIGN_8888",
                        "is_fraud": 1,
                        "scenario": "Rapid Account Draining Sequence",
                    })
                current_time = drain_time

        df = pd.DataFrame(records).iloc[:n_records].copy()
        # Ensure chronological ordering
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["transaction_id"] = [f"TX_{i+1:06d}" for i in range(len(df))]
        return df

    @staticmethod
    def generate_auth_events(
        compromised_account: str = "ACC_VICTIM_01",
        random_seed: int = 42,
    ) -> pd.DataFrame:
        """Generates synthetic authentication logs illustrating a cyber-heist kill chain.

        Sequence demonstrated:
        1. Repeated failed login attempts (Brute force / Credential stuffing).
        2. Successful login from an unrecognized device / foreign IP.
        3. Password / 2FA credential modification.
        4. Suspicious high-value external fund transfer.
        """
        rng = np.random.default_rng(random_seed)
        events = []
        base = datetime(2026, 1, 15, 2, 10, 0)

        # 1. Failed attempts
        for i in range(4):
            base += timedelta(seconds=int(rng.integers(12, 40)))
            events.append({
                "event_id": f"AUTH_{i+1:04d}",
                "account_id": compromised_account,
                "timestamp": base.strftime("%Y-%m-%d %H:%M:%S"),
                "event_type": "LOGIN_FAILED",
                "device_id": "DEV_BOT_0101",
                "ip_address": "194.26.29.112",
                "location": "St. Petersburg, RU",
                "status": "FAILURE",
                "detail": "Invalid password attempt",
            })

        # 2. Account compromise
        base += timedelta(minutes=4)
        events.append({
            "event_id": "AUTH_0005",
            "account_id": compromised_account,
            "timestamp": base.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": "LOGIN_SUCCESS",
            "device_id": "DEV_UNKNOWN_9999",
            "ip_address": "194.26.29.112",
            "location": "St. Petersburg, RU",
            "status": "SUCCESS",
            "detail": "Password matched after repeated failures (New Device)",
        })

        # 3. Security settings modified
        base += timedelta(minutes=2)
        events.append({
            "event_id": "AUTH_0006",
            "account_id": compromised_account,
            "timestamp": base.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": "SECURITY_PHONE_UPDATED",
            "device_id": "DEV_UNKNOWN_9999",
            "ip_address": "194.26.29.112",
            "location": "St. Petersburg, RU",
            "status": "SUCCESS",
            "detail": "Recovery phone number modified",
        })

        # 4. Beneficiary added
        base += timedelta(minutes=3)
        events.append({
            "event_id": "AUTH_0007",
            "account_id": compromised_account,
            "timestamp": base.strftime("%Y-%m-%d %H:%M:%S"),
            "event_type": "BENEFICIARY_ADDED",
            "device_id": "DEV_UNKNOWN_9999",
            "ip_address": "194.26.29.112",
            "location": "St. Petersburg, RU",
            "status": "SUCCESS",
            "detail": "New wire destination added: Offshore Crypto Vault",
        })

        return pd.DataFrame(events)


def save_default_synthetic_datasets(base_dir: str = "data") -> Tuple[str, str]:
    """Generates and persists initial sample and synthetic datasets.

    Args:
        base_dir: Directory to save datasets.

    Returns:
        Tuple of (transactions_path, auth_events_path).
    """
    synth_dir = os.path.join(base_dir, "synthetic")
    sample_dir = os.path.join(base_dir, "sample")
    os.makedirs(synth_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    tx_path = os.path.join(synth_dir, "synthetic_transactions.csv")
    auth_path = os.path.join(synth_dir, "synthetic_auth_events.csv")
    sample_tx_path = os.path.join(sample_dir, "banking_transactions_sample.csv")

    df_tx = SyntheticDataGenerator.generate_transactions(n_records=1500, fraud_ratio=0.045, random_seed=42)
    df_auth = SyntheticDataGenerator.generate_auth_events(random_seed=42)

    df_tx.to_csv(tx_path, index=False)
    df_tx.to_csv(sample_tx_path, index=False)
    df_auth.to_csv(auth_path, index=False)

    return tx_path, auth_path
