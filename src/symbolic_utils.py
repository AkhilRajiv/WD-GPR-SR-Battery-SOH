import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

# Ensure Julia depot and cache paths are strictly kept within the workspace on D: drive
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, '..'))
_JULIA_DIR = os.path.join(_PROJECT_DIR, '.julia')

os.environ['JULIA_DEPOT_PATH'] = _JULIA_DIR
os.environ['PYTHON_JULIAPKG_DIR'] = os.path.join(_JULIA_DIR, 'juliapkg')
os.environ['PYTHON_JULIACALL_DEPOT'] = _JULIA_DIR
os.environ['PYTHON_JULIACALL_HANDLE_SIGNALS'] = 'no'

def compute_inverse_variance_weights(
    sigma_pred: np.ndarray,
    clip_min_std: float = 1e-3
) -> np.ndarray:
    """
    Computes uncertainty weights as inverse predictive variance:
        w_i = 1 / sigma_i^2
    
    Weights are normalized so that mean(w) = 1.0. This prevents the overall
    loss scale from artificially blowing up while preserving the exact relative
    uncertainty penalties.
    """
    sig = np.asarray(sigma_pred, dtype=float).ravel()
    sig_clipped = np.maximum(sig, clip_min_std)
    var = sig_clipped ** 2

    raw_weights = 1.0 / var
    # Normalize to mean = 1.0
    normalized_weights = raw_weights / np.mean(raw_weights)
    return normalized_weights


def run_uncertainty_weighted_pysr(
    X_train: np.ndarray,
    y_residuals: np.ndarray,
    weights: Optional[np.ndarray],
    feature_names: list,
    random_seed: int = 42,
    niterations: int = 40,
    populations: int = 30,
    maxsize: int = 20,
    temp_equation_file: Optional[str] = None
) -> Tuple[Any, str, int, float]:
    """
    Runs PySR on GPR residuals using inverse variance weights.
    
    Parameters:
    -----------
    X_train: Array of shape (N, 4) in scaled feature space.
    y_residuals: Array of shape (N,) containing GPR residuals (e_i = y_true - y_hat_GPR).
    weights: Array of shape (N,) inverse-variance weights (or None for unweighted baseline).
    feature_names: List of 4 feature strings.
    random_seed: Deterministic integer seed.
    niterations: Number of PySR evolutionary cycles.
    populations: Number of parallel subpopulations.
    maxsize: Maximum equation tree complexity.
    
    Returns:
    --------
    (pysr_model, best_equation_str, complexity, training_loss)
    """
    # Import PySR dynamically after environment variables are set
    from pysr import PySRRegressor

    pysr_kwargs = {
        'niterations': niterations,
        'populations': populations,
        'maxsize': maxsize,
        'binary_operators': ["+", "-", "*", "/"],
        'unary_operators': ["square"],
        'elementwise_loss': "L2DistLoss()",
        'model_selection': "best",
        'random_state': random_seed,
        'deterministic': True,
        'parallelism': "serial",
        'progress': False,
        'input_stream': "devnull",
        'verbosity': 0
    }

    model = PySRRegressor(**pysr_kwargs)

    X_arr = np.asarray(X_train, dtype=np.float32)
    y_arr = np.asarray(y_residuals, dtype=np.float32).ravel()

    if weights is not None:
        w_arr = np.asarray(weights, dtype=np.float32).ravel()
        model.fit(X_arr, y_arr, weights=w_arr, variable_names=feature_names)
    else:
        model.fit(X_arr, y_arr, variable_names=feature_names)

    if temp_equation_file and hasattr(model, 'equations_') and model.equations_ is not None:
        try:
            os.makedirs(os.path.dirname(temp_equation_file), exist_ok=True)
            model.equations_.to_csv(temp_equation_file, index=False)
        except Exception as e:
            pass

    best_row = model.get_best()
    discovered_eqn = str(best_row['equation'])
    complexity = int(best_row['complexity'])
    training_loss = float(best_row['loss'])

    return model, discovered_eqn, complexity, training_loss


def predict_symbolic_correction(
    model: Any,
    X_test: np.ndarray
) -> np.ndarray:
    """
    Evaluates discovered symbolic equation r(x) on test features.
    """
    X_arr = np.asarray(X_test, dtype=np.float32)
    preds = np.asarray(model.predict(X_arr), dtype=float).ravel()
    
    if not np.isfinite(preds).all():
        # Fallback to 0.0 (no residual correction) for any pathological singularities (e.g. division by 0)
        preds = np.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)

    return preds
