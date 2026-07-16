# Architecture

## Frontend
- React
- TypeScript
- Vite
- Typed API client
- Responsive interface
- Loading and error states

## Backend
- FastAPI
- Pydantic schemas
- Modular routers
- Service layer for inference
- Pytest tests

## Machine learning

### Current
A deterministic keyword baseline provides an honest integration target.

### Next
1. Data validation
2. Train/validation/test split
3. TF-IDF features
4. Logistic-regression baseline
5. Macro F1 and per-class metrics
6. Error analysis
7. Saved preprocessing and model pipeline
8. MLflow experiment tracking

## Data layer
PostgreSQL will store tickets, predictions, human corrections, model versions, and feedback.

## Later LLM layer
- Retrieve relevant knowledge-base entries
- Draft a response
- Display supporting sources
- Require human review
- Log latency, errors, and feedback
