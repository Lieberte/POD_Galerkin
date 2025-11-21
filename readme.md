# POD-Based Surrogate Modeling for CFD Data

A Python framework for building reduced-order models of computational fluid dynamics simulations using Proper Orthogonal Decomposition (POD) and surrogate modeling techniques.

## Overview

This framework enables fast prediction of flow field solutions at new operating parameters by:
1. Extracting dominant flow structures via POD
2. Training surrogate models to map parameters to POD coefficients
3. Reconstructing full-field solutions from predicted coefficients

## Features

- Multi-variable flow field decomposition (velocity components, pressure)
- Flexible normalization strategies for improved numerical conditioning
- Multiple surrogate model options (RBF, GPR, Polynomial)
- Dual-space validation (normalized and physical domains)
- Per-variable error analysis and visualization
- Automated result archiving and metadata tracking

## Workflow

```
Training Data → POD Decomposition → Coefficient Extraction
↓
Surrogate Training
↓
New Parameters → Coefficient Prediction → Field Reconstruction → Validation
```

## Project Structure

```
├── src/
│   ├── pod_decomposition.py     # POD decomposition with SVD
│   ├── pod_surrogate.py         # Surrogate model implementations
│   └── pod_validation.py        # Validation and error metrics
├── database/
│   ├── csv/                     # Preprocessed training data
│   └── h5/                      # Validation snapshots
├── results/
│   └── {config_name}/
│       ├── normalized/          # Normalized space results
│       │   ├── POD/
│       │   ├── surrogate/
│       │   └── validation/
│       │       ├── normalized_space/
│       │       └── physical_space/
│       └── physical/            # Physical space results (if applicable)
└── main.py                      # Execution entry point
```

## Usage

```bash
python main.py
```

## Mathematical Formulation

**POD Decomposition:**
```
X = [x₁, x₂, ..., xₙ]  (snapshot matrix)
X_centered = X - x̄
U, Σ, Vᵀ = SVD(X_centered)
```

**Coefficient Projection:**
```
a = Uᵀ × (x - x̄)
```

**Field Reconstruction:**
```
x̂ = x̄ + U × â
```

where `â` is predicted by the surrogate model.

## Output Files

**POD Directory:**
- `modes.npy`: POD mode shapes
- `mean_field.npy`: Temporal mean field
- `singular_values.npy`: Energy spectrum
- `energy_distribution.png`: Mode energy visualization

**Surrogate Directory:**
- `surrogate_model.pkl`: Trained model
- `predictions_normalized.csv`: Test predictions (normalized)
- `predictions_physical.csv`: Test predictions (physical)

**Validation Directory:**
- `validation_errors.csv`: Error metrics per test case
- `predicted_fields.csv`: Full-field predictions
- `error_plot.png`: Error distribution plots
- `prediction_scatter.png`: Prediction vs truth scatter plots
- `{variable}/`: Per-variable error analysis

## Error Metrics

- **Relative L2**: `||x_true - x_pred|| / ||x_true||`
- **Max Absolute**: `max(|x_true - x_pred|)`
- **Mean Absolute**: `mean(|x_true - x_pred|)`
- **Max Relative**: `max(|x_true - x_pred| / |x_true|)`


## Notes

- Normalization is applied independently to each variable before POD
- Physical space validation requires storing both normalized and physical validation data
- Energy retention is automatically computed and visualized
- All matrix operations use column-major snapshot storage convention