# Machine-learning workspace

`train_baseline.py` trains a small TF-IDF + logistic-regression pipeline.

Important: `data/sample_tickets.csv` is a demonstration dataset created only
to exercise the code path. Its metrics must not be used as evidence of real
model quality.

Before making portfolio claims:

1. Select a legitimate dataset with a documented licence.
2. Define the label taxonomy.
3. Check class balance and leakage.
4. Create reproducible train/validation/test splits.
5. Compare against simple baselines.
6. Report per-class metrics and macro F1.
7. Perform error analysis.
