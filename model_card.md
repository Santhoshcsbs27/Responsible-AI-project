# Model Card: Responsible AI Demo Model

**Model type:** Logistic regression

**Intended Use:**
- Predicts a binary admission decision for a synthetic applicant dataset.
- Demonstrates responsible AI practices including fairness metrics, basic explainability, and model documentation.

**Primary Audience:** Developers exploring responsible AI patterns and building prototypes.

## Model Details

- **Input features:** age, gpa, study_hours, gender
- **Protected attribute (sensitive):** `gender` (0=female, 1=male)

## Fairness Considerations

This project includes a simple fairness metric computation:

- **Demographic Parity Difference**: Difference in positive prediction rates between groups.
- **Equal Opportunity Difference**: Difference in true positive rates.

The underlying synthetic dataset is intentionally biased to illustrate how
model outputs can differ across protected groups.

## Limitations

- This is a toy model trained on a synthetic dataset; it is not suitable for
  real-world decision making.
- Explanation is based on model coefficients and does not represent a full
  counterfactual or SHAP-style explanation.
- More robust audits should include cross-validation, calibration analysis, and
  human-in-the-loop review.

## How to Use

Start the Flask API and interact with endpoints:

- `GET /health` - health check
- `POST /predict` - get a prediction given input features
- `POST /explain` - get a feature contribution breakdown
- `GET /fairness` - view fairness metrics on a sample dataset
