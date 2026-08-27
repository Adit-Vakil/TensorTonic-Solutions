import numpy as np

def minmax_scale(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    X = np.array(X, dtype=float)
    X_min = np.min(X, axis=axis, keepdims=True)
    X_max = np.max(X, axis=axis, keepdims=True)
    range_ = X_max - X_min
    return (X - X_min) / (range_ + eps)