import numpy as np

def percentiles(x: list, q: list) -> np.ndarray:
    xs = np.sort(np.asarray(x, dtype = float))
    n = len(xs)
    q = np.asarray(q, dtype = float)
    r = q / 100 * (n - 1)
    l = np.floor(r).astype(int)
    u = np.ceil(r).astype(int)
    w = r - l
    return (1 - w) * xs[l] + w * xs[u]