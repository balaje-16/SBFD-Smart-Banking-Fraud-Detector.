"""Integration tests for the complete FraudDetectionPipeline across multiple dataset schemas."""

import os
import pytest
import numpy as np
import pandas as pd

from src.data_loader import SyntheticDataGenerator, DatasetAdapter
from src.pipeline import FraudDetectionPipeline


def test_pipeline_synthetic_execution(tmp_path):
    df_synth = SyntheticDataGenerator.generate_transactions(n_records=200, random_seed=42)
    alerts_path = os.path.join(tmp_path, "pipeline_alerts.json")
    model_path = os.path.join(tmp_path, "pipe_model.joblib")

    pipeline = FraudDetectionPipeline(
        contamination=0.04,
        alerts_storage_path=alerts_path,
    )
    result = pipeline.run(df_synth, train_new_model=True, model_checkpoint_path=model_path)

    assert len(result.features_dataframe) == len(result.cleaned_dataframe)
    assert "anomaly_score" in result.features_dataframe.columns
    assert "overall_risk_score" in result.features_dataframe.columns
    assert "risk_category" in result.features_dataframe.columns
    assert len(result.evaluation_results) == len(result.features_dataframe)
    assert result.metrics is not None
    assert result.metrics.has_ground_truth is True
    assert os.path.exists(model_path)


def test_pipeline_credit_card_format(tmp_path):
    # Simulated Kaggle Credit Card dataset (Time, V1..V5, Amount, Class)
    raw_cc = pd.DataFrame({
        "Time": list(range(50)),
        "V1": np.random.randn(50),
        "V2": np.random.randn(50),
        "V3": np.random.randn(50),
        "Amount": [10.0 + i * 2.0 for i in range(50)],
        "Class": [1 if i % 10 == 0 else 0 for i in range(50)],
    })

    adapted, meta = DatasetAdapter.adapt(raw_cc)
    alerts_path = os.path.join(tmp_path, "cc_alerts.json")

    pipeline = FraudDetectionPipeline(alerts_storage_path=alerts_path)
    result = pipeline.run(adapted, train_new_model=True)

    # Profiling should be empty because CC dataset has no account IDs
    assert len(result.account_profiles) == 0
    assert "anomaly_score" in result.features_dataframe.columns
    assert len(result.evaluation_results) == 50


def test_pipeline_paysim_format(tmp_path):
    # Simulated PaySim dataset
    raw_ps = pd.DataFrame({
        "step": [1] * 20 + [2] * 20,
        "type": ["TRANSFER"] * 40,
        "amount": [50.0 + i * 15.0 for i in range(40)],
        "nameOrig": [f"C_ORIG_{i % 5}" for i in range(40)],
        "oldbalanceOrg": [1000.0] * 40,
        "newbalanceOrig": [900.0] * 40,
        "nameDest": [f"M_DEST_{i % 4}" for i in range(40)],
        "oldbalanceDest": [0.0] * 40,
        "newbalanceDest": [0.0] * 40,
        "isFraud": [1 if i == 39 else 0 for i in range(40)],
    })

    adapted, meta = DatasetAdapter.adapt(raw_ps)
    alerts_path = os.path.join(tmp_path, "ps_alerts.json")

    pipeline = FraudDetectionPipeline(alerts_storage_path=alerts_path)
    result = pipeline.run(adapted, train_new_model=True)

    assert meta["has_account_id"] is True
    assert len(result.account_profiles) > 0
    assert len(result.features_dataframe) == 40
