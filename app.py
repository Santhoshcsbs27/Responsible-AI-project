"""A small Flask app demonstrating responsible AI best practices."""

from __future__ import annotations

import logging
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from model import ResponsibleAIModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    app = Flask(__name__)

    model = ResponsibleAIModel(retrain=False)

    @app.route("/health", methods=["GET"])
    def health() -> tuple[dict, int]:
        return {"status": "ok", "model_version": model.metadata.get("version")}, 200

    @app.route("/predict", methods=["POST"])
    def predict() -> tuple[dict, int]:
        payload = request.get_json(force=True)
        if payload is None:
            return {"error": "Invalid JSON payload."}, 400
        result = model.predict(payload)
        return result, 200

    @app.route("/explain", methods=["POST"])
    def explain() -> tuple[dict, int]:
        payload = request.get_json(force=True)
        if payload is None:
            return {"error": "Invalid JSON payload."}, 400
        result = model.explain(payload)
        return result, 200

    @app.route("/fairness", methods=["GET"])
    def fairness() -> tuple[dict, int]:
        metrics = model.fairness_metrics()
        return {"fairness_metrics": metrics}, 200

    @app.route("/", methods=["GET"])
    def index() -> str:
        return render_template("index.html"), 200

    @app.route("/model-card", methods=["GET"])
    def model_card() -> tuple[str, int]:
        card_path = Path(__file__).parent / "model_card.md"
        return card_path.read_text(encoding="utf-8"), 200

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000)
