import numpy as np

def t_test_one_sample(x: list, mu0: float) -> float:
    x = np.array(x, dtype = float)
    n = x.size
    xbar = x.mean()
    s = x.std(ddof = 1)
    diff = xbar - mu0

    if s == 0:
        if diff == 0:
            return 0.0
        return float(np.sign(diff) * np.inf)

    t = diff / (s / np.sqrt(n))
    return float(t)