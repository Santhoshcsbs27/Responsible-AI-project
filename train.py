"""A simple training script for the Responsible AI demo.

This script is intended to be used during local development to retrain the
model and refresh the persisted artifact in `models/model.pkl`.

Usage:
    python train.py
"""

from model import ResponsibleAIModel


if __name__ == "__main__":
    model = ResponsibleAIModel(retrain=True)
    print("Model training complete. Model saved to models/model.pkl")
