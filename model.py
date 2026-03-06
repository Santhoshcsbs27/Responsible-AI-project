"""Responsible AI model utilities.

This module contains a simple logistic classification model that is trained on a
synthetic dataset. It exposes helper functions to compute fairness metrics and
basic explanations.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import generate_synthetic_admission_data, get_feature_columns


logger = logging.getLogger(__name__)


@dataclass
class FairnessMetrics:
    demographic_parity_difference: float
    equal_opportunity_difference: float

    def to_dict(self) -> dict:
        return {
            "demographic_parity_difference": self.demographic_parity_difference,
            "equal_opportunity_difference": self.equal_opportunity_difference,
        }


class ResponsibleAIModel:
    """A small responsible AI model with fairness and explainability helpers."""

    MODEL_PATH = Path("models/model.pkl")
    METADATA_PATH = Path("models/model-metadata.json")

    def __init__(self, retrain: bool = False):
        self.model: Pipeline | None = None
        self._metadata: dict | None = None

        if retrain or not self.MODEL_PATH.exists():
            logger.info("Training new model (retrain=%s)", retrain)
            self.train()
        else:
            self.load()

    def train(self) -> None:
        """Train a new model and persist it to disk."""

        df = generate_synthetic_admission_data(n_samples=12_000, random_state=42)
        feature_cols = get_feature_columns()
        X = df[feature_cols]
        y = df["admitted"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )

        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(solver="liblinear", random_state=42),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)

        self.model = pipeline

        self.MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)
        joblib.dump(self.model, self.MODEL_PATH)

        self._metadata = {
            "version": "0.1.0",
            "trained_at": pd.Timestamp.now().isoformat(),
            "feature_columns": feature_cols,
            "description": "Logistic regression trained on synthetic admission data.",
        }
        with open(self.METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(self._metadata, f, indent=2)

        logger.info("Model trained and saved to %s", self.MODEL_PATH)

    def load(self) -> None:
        """Load the model and metadata from disk."""

        self.model = joblib.load(self.MODEL_PATH)
        self._metadata = json.loads(self.METADATA_PATH.read_text(encoding="utf-8"))
        logger.info("Loaded model from %s", self.MODEL_PATH)

    @property
    def metadata(self) -> dict:
        if self._metadata is None:
            self._metadata = {}
        return self._metadata

    def predict(self, payload: dict) -> dict:
        """Predict label and probability for a single instance."""

        feature_cols = self.metadata.get("feature_columns", get_feature_columns())
        X = np.array([[payload.get(col) for col in feature_cols]])
        proba = self.model.predict_proba(X)[0, 1]
        label = int(self.model.predict(X)[0])

        return {
            "prediction": label,
            "probability": float(proba),
            "feature_names": feature_cols,
            "feature_values": payload,
        }

    def explain(self, payload: dict) -> dict:
        """Return feature importance explanation for a single prediction."""

        feature_cols = self.metadata.get("feature_columns", get_feature_columns())
        coef = self.model.named_steps["clf"].coef_[0]
        intercept = float(self.model.named_steps["clf"].intercept_[0])

        # It is not a full SHAP explanation but provides a simple coefficient view.
        contributions = [float(coef[i] * payload.get(feature_cols[i], 0)) for i in range(len(feature_cols))]

        return {
            "intercept": intercept,
            "coefficients": dict(zip(feature_cols, coef.tolist())),
            "feature_contributions": dict(zip(feature_cols, contributions)),
        }

    def fairness_metrics(self, df: pd.DataFrame | None = None) -> dict:
        """Compute simple fairness metrics on a dataset.

        Demographic parity difference = P(pred=1 | gender=1) - P(pred=1 | gender=0)
        Equal opportunity difference = TPR(gender=1) - TPR(gender=0)
        """

        if df is None:
            df = generate_synthetic_admission_data(n_samples=8_000, random_state=123)

        feature_cols = self.metadata.get("feature_columns", get_feature_columns())
        X = df[feature_cols]
        y = df["admitted"]

        preds = self.model.predict(X)
        df_eval = df.copy()
        df_eval["predicted"] = preds

        def parity(group_val: int) -> float:
            group = df_eval[df_eval["gender"] == group_val]
            if len(group) == 0:
                return 0.0
            return float(group["predicted"].mean())

        def true_positive_rate(group_val: int) -> float:
            group = df_eval[df_eval["gender"] == group_val]
            positives = group[group["admitted"] == 1]
            if len(positives) == 0:
                return 0.0
            return float((positives["predicted"] == 1).mean())

        dp_male = parity(1)
        dp_female = parity(0)
        tpr_male = true_positive_rate(1)
        tpr_female = true_positive_rate(0)

        metrics = FairnessMetrics(
            demographic_parity_difference=dp_male - dp_female,
            equal_opportunity_difference=tpr_male - tpr_female,
        )
        return metrics.to_dict()
