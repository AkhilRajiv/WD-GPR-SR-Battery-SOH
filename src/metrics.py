import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from typing import Dict, Any, Union

def compute_regression_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    sigma_pred: Union[np.ndarray, list, None] = None
) -> Dict[str, float]:
    """
    Computes both point accuracy metrics (MAE, MSE, RMSE, R2, MAPE)
    and probabilistic uncertainty calibration metrics (NLL, 95% PICP, MPIW).
    
    Parameters:
    -----------
    y_true: Array-like of ground truth target values (original scale, e.g., SOH %)
    y_pred: Array-like of predicted mean values (original scale)
    sigma_pred: Optional array-like of predictive standard deviations (original scale)
    
    Returns:
    --------
    dict of evaluated metric scores.
    """
    y_t = np.asarray(y_true, dtype=float).ravel()
    y_p = np.asarray(y_pred, dtype=float).ravel()

    if len(y_t) != len(y_p):
        raise ValueError(f"Length mismatch: y_true ({len(y_t)}) vs y_pred ({len(y_p)})")

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))
    mape = float(mean_absolute_percentage_error(y_t, y_p) * 100.0)

    results = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2,
        'MAPE(%)': mape
    }

    if sigma_pred is not None:
        sig = np.asarray(sigma_pred, dtype=float).ravel()
        if len(sig) != len(y_t):
            raise ValueError(f"Length mismatch: sigma_pred ({len(sig)}) vs y_true ({len(y_t)})")

        # Clamp standard deviations to small positive epsilon for numerical stability
        eps = 1e-6
        sig = np.maximum(sig, eps)
        var = sig ** 2

        # 1. Negative Log-Likelihood (Gaussian predictive distribution)
        # NLL = 0.5 * mean( log(2 * pi * var) + (y - mu)^2 / var )
        nll = float(0.5 * np.mean(np.log(2.0 * np.pi * var) + ((y_t - y_p) ** 2) / var))

        # 2. 95% Prediction Interval Coverage Probability (PICP)
        # z_crit = 1.95996
        z_crit = 1.95996398454005423552
        lower_bound = y_p - z_crit * sig
        upper_bound = y_p + z_crit * sig
        inside = (y_t >= lower_bound) & (y_t <= upper_bound)
        picp = float(np.mean(inside) * 100.0)

        # 3. Mean Prediction Interval Width (MPIW)
        # MPIW = mean( 2 * z_crit * sig )
        mpiw = float(np.mean(upper_bound - lower_bound))

        results.update({
            'NLL': nll,
            'PICP(%)': picp,
            'MPIW': mpiw
        })

    return results
