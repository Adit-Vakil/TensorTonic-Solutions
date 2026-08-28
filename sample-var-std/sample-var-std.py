import numpy as np

def sample_var_std(x: list) -> dict:
    arr = np.array(x, dtype = float)
    var = np.var(arr, ddof = 1)
    std = np.sqrt(var)
    return {"variance": float(var), "standard_deviation": float(std)}