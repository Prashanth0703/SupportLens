# SupportLens ML Milestone 1

This patch adds:

- Bitext customer-support dataset download
- Cleaning and stratified 70/15/15 splits
- Majority-class validation baseline
- TF-IDF + logistic-regression classifier
- Accuracy, macro F1, weighted F1, and per-class report
- Normalised confusion matrix
- MLflow experiment tracking
- Offline unit tests
- GitHub Actions ML workflow

## Apply the patch

Copy the `ml` folder into the repository, replacing the current starter ML files.
Copy `.github/workflows/ml-ci.yml` into the repository.

Add these lines to the repository `.gitignore`:

```gitignore
data/processed/
ml/artifacts/
ml/mlruns/
```

## Run on Windows PowerShell

```powershell
cd ml
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pytest -q
python prepare_dataset.py
python train_baseline.py
mlflow server --port 5000
```

Open:

`http://127.0.0.1:5000`
