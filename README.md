# Feature Selection with Binary PSO

Selecting the most useful input features for a regression model with **binary particle swarm optimization (PSO)**, implemented from scratch in Python and NumPy.

## Problem

An industrial dataset with **50 features** and about **2,900 samples**. The goal is to find the subset of features that best predicts the target with a linear regression model.

## Approach

- **Binary PSO:** each particle is a 0/1 vector over the 50 features (1 = feature selected). Velocities are mapped to selection probabilities with a sigmoid function.
- **Parameters:** 50 particles, 150 iterations, inertia weight decreasing linearly from 0.9 to 0.4, c1 = c2 = 1.5
- **Fitness:** test-set MSE of a linear regression model. The regression coefficients are solved with QR decomposition for numerical stability.
- **Search:** run PSO for every subset size n = 1 to 50 and keep the best
- **Baseline:** compared against choosing features by their correlation with the target (`Correlation.py`)

## Results

| Number of features | Best test MSE |
| --- | --- |
| 1 | 0.204 |
| 5 | 0.120 |
| 39 (best overall) | **0.0975** |

Full log: `data_text/Final_Parameter_Outcome_Best_39.txt`. Charts, including the correlation-based comparison, are in `picture/`.

## How to run

```bash
pip install -r requirements.txt
python CleanData.py        # data/industrial_dataset.csv -> data/industrial_data.csv
python ExtractTestData.py  # create the test sample data/sampled_data.csv
python Correlation.py      # correlation-based baseline
python PSO.py              # run PSO for n = 1..50 and print the best subset
```

The cleaned data and test sample are already included in `data/`, so you can run `PSO.py` directly. A full run over n = 1..50 took about 30 minutes on my machine.

## Files

| File | Purpose |
| --- | --- |
| `PSO.py` | Binary PSO and the regression fitness function |
| `CleanData.py` | Removes the extra index column |
| `ExtractTestData.py` | Samples the test set |
| `Correlation.py` | Correlation-based feature ranking |
| `PSO_old.py` | Earlier version, kept for reference |
