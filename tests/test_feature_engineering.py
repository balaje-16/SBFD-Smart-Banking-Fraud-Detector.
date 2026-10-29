"""Tests for causal feature engineering and cold-start fallback handling."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.feature_engineering import FeatureEngineer


def test_feature_engineering_causality():
    # Construct sequence where account A starts with small transactions, then does a large one
    t0 = datetime(2026, 1, 1, 10, 0, 0)
    df = pd.DataFrame({
        "transaction_id": ["TX_1", "TX_2", "TX_3", "TX_4", "TX_5"],
        "account_id": ["ACC_A", "ACC_A", "ACC_A", "ACC_A", "ACC_A"],
        "timestamp": [t0 + timedelta(hours=i * 2) for i in range(5)],
        "amount": [50.0, 50.0, 50.0, 50.0, 5000.0],  # 5th is huge jump
    })

    fe = FeatureEngineer(min_history_records=3)
    feat_df, feat_names = fe.compute_features(df, is_training=True)

    # First 3 are cold-start
    assert feat_df["feature_is_cold_start"].iloc[0] == 1.0
    assert feat_df["feature_is_cold_start"].iloc[1] == 1.0
    assert feat_df["feature_is_cold_start"].iloc[2] == 1.0

    # 4th transaction has 3 historical transactions (mean=50.0, ratio=1.0)
    assert feat_df["feature_is_cold_start"].iloc[3] == 0.0
    assert np.isclose(feat_df["feature_amount_ratio_hist"].iloc[3], 1.0, atol=0.1)

    # 5th transaction ($5000) has historical past mean = 50.0
    # Crucial test: future $5000 did NOT alter the 4th transaction's mean!
    assert feat_df["feature_amount_ratio_hist"].iloc[4] > 50.0
    # Frequency counts strictly prior within 24h
    assert feat_df["feature_freq_24h"].iloc[4] >= 4.0


def test_feature_engineering_without_account_ids():
    # E.g. Credit Card dataset without account identifiers
    t0 = datetime(2026, 1, 1, 10, 0, 0)
    df = pd.DataFrame({
        "transaction_id": ["TX_1", "TX_2", "TX_3"],
        "account_id": [None, None, None],
        "timestamp": [t0 + timedelta(minutes=i * 10) for i in range(3)],
        "amount": [120.0, 85.0, 990.0],
        "V1": [-0.5, 0.2, 1.4],
        "V2": [0.1, -0.9, 0.3],
    })

    fe = FeatureEngineer()
    feat_df, feat_names = fe.compute_features(df, is_training=True)

    assert "V1" in feat_names
    assert "V2" in feat_names
    assert "feature_amount" in feat_names
    assert feat_df["feature_is_cold_start"].iloc[0] == 1.0


def test_temporal_features():
    # Night transaction vs day transaction
    df = pd.DataFrame({
        "transaction_id": ["TX_DAY", "TX_NIGHT"],
        "account_id": ["A1", "A2"],
        "timestamp": [
            pd.Timestamp("2026-01-01 14:00:00"),  # 2 PM
            pd.Timestamp("2026-01-01 02:30:00"),  # 2:30 AM (Night)
        ],
        "amount": [25.0, 40.0],
    })

    fe = FeatureEngineer()
    feat_df, _ = fe.compute_features(df)

    assert feat_df[feat_df["transaction_id"] == "TX_DAY"]["feature_is_night"].iloc[0] == 0.0
    assert feat_df[feat_df["transaction_id"] == "TX_NIGHT"]["feature_is_night"].iloc[0] == 1.0
