# Responsible AI Project (Flask)

A small demo project showcasing responsible AI practices using a Flask API.

## 🚀 Features

- ✅ Flask REST API for predictions and explanations
- ✅ Synthetic dataset with a protected attribute (`gender`)
- ✅ Fairness metrics (demographic parity, equal opportunity)
- ✅ Basic explainability via feature contributions
- ✅ Model card and documentation

## 🧰 Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## ▶️ Run the API

```bash
python app.py
```

The server will start on `http://0.0.0.0:8000`.

## 🧪 Example Requests

### Health check

```bash
curl http://localhost:8000/health
```

### Predict

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"age": 22, "gpa": 3.2, "study_hours": 10, "gender": 0}'
```

### Explain

```bash
curl -X POST http://localhost:8000/explain -H "Content-Type: application/json" \
  -d '{"age": 22, "gpa": 3.2, "study_hours": 10, "gender": 0}'
```

### Fairness metrics

```bash
curl http://localhost:8000/fairness
```

## 🧪 Retrain the model

```bash
python train.py
```
