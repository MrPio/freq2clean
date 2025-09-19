import numpy as np


def ale(x: np.ndarray, y: np.ndarray) -> float:
    """Compute absolute lifetime error (aka MAPE)"""
    # Flatten arrays
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()

    mask = y != 0
    ale = np.mean(np.abs((x[mask] - y[mask]) / y[mask])) * 100
    return ale
