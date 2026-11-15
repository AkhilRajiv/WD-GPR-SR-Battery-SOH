
# Battery State-of-Health Prediction using WD-GPR-SR

## Project Overview

This project investigates lithium-ion battery State-of-Health (SOH)
prediction using cycle-level measurements from the CALCE battery dataset.

The proposed approach combines:
- Wide-and-Deep Gaussian Process Regression
- Sparse Variational Gaussian Process
- Uncertainty-Weighted Symbolic Residual Correction

## Dataset

CALCE lithium-ion battery ageing data.

Cells used:
- CS2_35
- CS2_36
- CS2_37
- CS2_38

## Methodology

1. Dataset preparation and preprocessing
2. Health indicator extraction
3. Baseline model evaluation
4. WD-GPR-SR model development
5. Ablation study
6. Explainability analysis

## Evaluation

Four-fold Leave-One-Cell-Out evaluation.

Metrics include MAE, RMSE, R², MAPE, NLL,
95% Prediction Interval Coverage and Mean Prediction
Interval Width.

## Repository Structure

- notebooks/: Consolidated project notebook
- src/: Supporting Python source code
- results/: Selected results and figures
- report/: Final project report
- presentation/: Project presentation

## Reproducibility

Install the required dependencies using:

    pip install -r requirements.txt

Download and place the CALCE dataset in the
required local data directory.

Follow the notebook instructions for preprocessing,
model execution and evaluation.

## Author

Satya Akhil Rajiv Bodapati

Roll Number: 2401AI32
