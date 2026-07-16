# SupportLens

SupportLens is an end-to-end customer-support intelligence platform built as a portfolio project.

## Planned capabilities

- Ticket classification and priority prediction
- Duplicate-ticket detection
- Knowledge-base retrieval
- LLM-assisted response drafting with source references
- Human review and feedback
- Operational and model-performance dashboards
- Experiment tracking and model versioning

## Current milestone: Vertical Slice v0.1

The first milestone includes:

- React + TypeScript frontend
- FastAPI backend
- Transparent keyword-based baseline classifier
- Sample ticket dataset
- Starter scikit-learn training pipeline
- Backend tests
- Docker configuration
- Architecture and roadmap documents

The current classifier is deliberately labelled `keyword_baseline_v0`. It is not a trained ML model. It provides a working frontend/backend integration point before the proper ML pipeline is complete.

## Quick start

### Backend

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn app.main:app --reload
```

API: `http://localhost:8000`  
Docs: `http://localhost:8000/docs`

Run tests:

```bash
pytest
```

### Frontend

Use Node.js 20.19+ or another currently supported release.

```bash
cd frontend
npm install
npm run dev
```

Frontend: `http://localhost:5173`

### Docker

```bash
docker compose up --build
```

## ML baseline

The included CSV is only a demonstration dataset and is not suitable for making performance claims.

```bash
cd ml
pip install -r requirements.txt
python train_baseline.py
```

See `docs/ROADMAP.md` for the implementation plan.
