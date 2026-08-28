import numpy as np

def zscore_standardize(X: list, axis: int = 0, eps: float = 1e-12) -> np.ndarray:
    X = np.array(X, dtype = float)
    mew = np.mean(X, axis = axis, keepdims = True)
    dev = np.std(X, axis = axis, keepdims = True)
    safe_dev = np.where(dev <= eps, 1.0, dev)
    z = (X-mew) / safe_dev
    z = np.where(dev <= eps, 0.0, z)
    return z