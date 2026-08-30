import numpy as np

def chi2_independence(C: list) -> dict:
    O = np.array(C, dtype = float)
    R = O.sum(axis = 1, keepdims = True)
    Cj = O.sum(axis = 0, keepdims = True)
    N = O.sum()
    expected = R @ Cj / N
    chi2 = float(np.sum((O - expected) ** 2 / expected))
    return {"chi2": chi2, "expected": expected}