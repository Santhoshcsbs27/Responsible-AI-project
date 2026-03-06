"""Synthetic dataset creation for responsible AI examples."""

from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_admission_data(
    *,
    n_samples: int = 10_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate a simple synthetic dataset with a sensitive attribute.

    The dataset is designed to simulate a binary admission decision with
    a protected attribute (gender) that impacts label distribution.

    Returns:
        A DataFrame with columns: [age, gpa, study_hours, gender, admitted]
    """

    rng = np.random.default_rng(random_state)

    # Features
    age = rng.normal(loc=22, scale=2.5, size=n_samples).clip(17, 30)
    gpa = rng.normal(loc=3.0, scale=0.5, size=n_samples).clip(0.0, 4.0)
    study_hours = rng.normal(loc=10, scale=4, size=n_samples).clip(0, 30)

    # Sensitive attribute: 0 = female, 1 = male
    gender = rng.choice([0, 1], size=n_samples, p=[0.5, 0.5])

    # Base score
    score = 0.5 * gpa + 0.03 * age + 0.02 * study_hours

    # Introduce a controlled bias for the male group
    score += np.where(gender == 1, 0.15, -0.05)

    # Convert to probability via logistic function
    proba = 1 / (1 + np.exp(- (score - 2.5)))

    admitted = rng.binomial(1, proba)

    return pd.DataFrame(
        {
            "age": age,
            "gpa": gpa,
            "study_hours": study_hours,
            "gender": gender,
            "admitted": admitted,
        }
    )


def get_feature_columns() -> list[str]:
    """Return the ordered list of input feature names."""

    return ["age", "gpa", "study_hours", "gender"]
