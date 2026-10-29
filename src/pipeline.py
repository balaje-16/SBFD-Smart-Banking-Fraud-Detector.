"""End-to-end detection pipeline orchestrator.

Coordinates dataset loading, validation, feature extraction, Isolation Forest inference,
behavioural profiling, sequence rule evaluation, contextual risk scoring, and alert generation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.alert_manager import AlertManager, FraudAlert
from src.anomaly_scoring import AnomalyScoreNormalizer
from src.behavioural_analysis import AccountProfile, BehaviouralAnalyzer
from src.data_loader import DatasetAdapter, SyntheticDataGenerator, load_csv_safely
from src.evaluation import EvaluationMetrics, ModelEvaluator
from src.feature_engineering import FeatureEngineer
from src.model import IsolationForestAnomalyDetector
from src.preprocessing import DataPreprocessor, DataQualityReport
from src.risk_engine import ContextualRiskEngine, RiskEvaluationResult, RiskWeights
from src.sequence_detection import SequenceDetector, SequenceViolation


@dataclass
class PipelineExecutionResult:
    """Encapsulates the full output of an executed detection run."""

    raw_dataframe: pd.DataFrame
    cleaned_dataframe: pd.DataFrame
    features_dataframe: pd.DataFrame
    quality_report: DataQualityReport
    account_profiles: Dict[str, AccountProfile]
    sequence_violations: List[SequenceViolation]
    evaluation_results: List[RiskEvaluationResult]
    alerts_created: List[FraudAlert]
    metrics: Optional[EvaluationMetrics] = None
    unlabeled_summary: Optional[Dict[str, Any]] = None
    feature_names: List[str] = field(default_factory=list)


class FraudDetectionPipeline:
    """High-level pipeline coordinator."""

    def __init__(
        self,
        contamination: float = 0.02,
        risk_weights: Optional[RiskWeights] = None,
        threshold_medium: float = 40.0,
        threshold_high: float = 70.0,
        alerts_storage_path: str = "data/alerts.json",
    ):
        """Initialize pipeline components."""
        self.contamination = contamination
        self.risk_weights = risk_weights or RiskWeights()
        self.threshold_medium = threshold_medium
        self.threshold_high = threshold_high
        self.alerts_storage_path = alerts_storage_path

        self.preprocessor = DataPreprocessor()
        self.feature_engineer = FeatureEngineer()
        self.model = IsolationForestAnomalyDetector(contamination=contamination)
        self.normalizer = AnomalyScoreNormalizer()
        self.behavioural_analyzer = BehaviouralAnalyzer()
        self.sequence_detector = SequenceDetector()
        self.risk_engine = ContextualRiskEngine(
            weights=self.risk_weights,
            threshold_medium=threshold_medium,
            threshold_high=threshold_high,
        )
        self.alert_manager = AlertManager(storage_path=alerts_storage_path)

    def run(
        self,
        df_input: pd.DataFrame,
        train_new_model: bool = True,
        model_checkpoint_path: Optional[str] = None,
    ) -> PipelineExecutionResult:
        """Executes end-to-end detection on the input DataFrame.

        Args:
            df_input: Adapted or raw DataFrame.
            train_new_model: Whether to fit Isolation Forest on this data or use pre-trained.
            model_checkpoint_path: Path to load or save fitted model.

        Returns:
            PipelineExecutionResult containing enriched tables and alert outputs.
        """
        # 1. Validation and cleaning
        clean_df, quality_report = self.preprocessor.validate_and_clean(df_input)

        # 2. Chronological feature engineering
        feat_df, feature_names = self.feature_engineer.compute_features(clean_df, is_training=train_new_model)

        # 3. Scale features
        X_scaled = self.preprocessor.encode_and_scale(feat_df, feature_names, is_training=train_new_model)

        # 4. Model Training / Inference
        if train_new_model:
            self.model = IsolationForestAnomalyDetector(contamination=self.contamination)
            self.model.fit(X_scaled, feature_names)
            raw_scores = self.model.predict_raw_scores(X_scaled)
            self.normalizer.calibrate(raw_scores)
            if model_checkpoint_path:
                self.model.save(model_checkpoint_path)
        else:
            if model_checkpoint_path and os.path.exists(model_checkpoint_path):
                self.model = IsolationForestAnomalyDetector.load(model_checkpoint_path)
            raw_scores = self.model.predict_raw_scores(X_scaled)

        # 5. Normalized Anomaly Score
        norm_scores = self.normalizer.normalize(raw_scores)
        feat_df["anomaly_score"] = np.round(norm_scores, 1)

        # 6. Behavioural Profiling
        has_accounts = "account_id" in feat_df.columns and feat_df["account_id"].notna().any()
        if has_accounts:
            profiles = self.behavioural_analyzer.build_profiles(feat_df)
            baselines = {acc: p.mean_amount for acc, p in profiles.items()}
        else:
            profiles = {}
            baselines = {}

        # 7. Sequence Rule Violations
        sequence_violations = self.sequence_detector.detect_transaction_sequences(feat_df, account_baselines=baselines)
        seq_violation_map: Dict[str, List[SequenceViolation]] = {}
        for v in sequence_violations:
            seq_violation_map.setdefault(v.trigger_transaction_id, []).append(v)

        # 8. Contextual Risk Fusion
        eval_results: List[RiskEvaluationResult] = []
        overall_scores = []
        risk_categories = []
        explanations = []

        for idx, row in feat_df.iterrows():
            tx_id = str(row["transaction_id"])
            acc_id = str(row["account_id"]) if pd.notna(row.get("account_id")) else None
            amt = float(row["amount"])
            ml_score = float(row["anomaly_score"])

            # Account deviation
            dev_info = self.behavioural_analyzer.evaluate_transaction_deviation(acc_id, amt)

            # Sequence violations for this transaction
            v_list = seq_violation_map.get(tx_id, [])
            v_score = max([v.risk_contribution for v in v_list], default=0.0)
            v_notes = [v.description for v in v_list]

            result = self.risk_engine.evaluate_transaction(
                transaction_id=tx_id,
                account_id=acc_id,
                amount=amt,
                anomaly_score=ml_score,
                deviation_severity=float(dev_info["deviation_severity"]),
                amount_ratio=float(dev_info["amount_ratio"]),
                tx_velocity=float(row.get("feature_tx_velocity", 0.0)),
                freq_1h=float(row.get("feature_freq_1h", 1.0)),
                is_night=bool(row.get("feature_is_night", 0.0)),
                is_weekend=bool(row.get("feature_is_weekend", 0.0)),
                sequence_violation_score=v_score,
                sequence_notes=v_notes,
                is_cold_start=bool(row.get("feature_is_cold_start", 0.0)),
            )

            eval_results.append(result)
            overall_scores.append(result.overall_risk_score)
            risk_categories.append(result.risk_category)
            explanations.append(" | ".join(result.reasons))

        feat_df["overall_risk_score"] = overall_scores
        feat_df["risk_category"] = risk_categories
        feat_df["explanation"] = explanations

        # 9. Alert Generation & Persistence
        alerts_created = self.alert_manager.generate_alerts_from_evaluations(eval_results, auto_persist=True)

        # 10. Evaluation (Ground Truth check)
        metrics: Optional[EvaluationMetrics] = None
        unlabeled_summary: Optional[Dict[str, Any]] = None

        has_ground_truth = "is_fraud" in feat_df.columns and feat_df["is_fraud"].notna().any()
        if has_ground_truth:
            y_true = feat_df["is_fraud"].fillna(0).astype(int).values
            # Run evaluation at the high risk threshold
            metrics = ModelEvaluator.evaluate_predictions(
                y_true=y_true,
                risk_scores=np.array(overall_scores),
                threshold=self.threshold_high,
            )
        else:
            unlabeled_summary = ModelEvaluator.summarize_unlabeled_anomalies(np.array(overall_scores))

        return PipelineExecutionResult(
            raw_dataframe=df_input,
            cleaned_dataframe=clean_df,
            features_dataframe=feat_df,
            quality_report=quality_report,
            account_profiles=profiles,
            sequence_violations=sequence_violations,
            evaluation_results=eval_results,
            alerts_created=alerts_created,
            metrics=metrics,
            unlabeled_summary=unlabeled_summary,
            feature_names=feature_names,
        )
