import numpy as np

def r2_score(y_true: list, y_pred: list) -> float:
    y_true = np.asarray(y_true, dtype = float)
    y_pred = np.asarray(y_pred, dtype = float)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)

    if ss_tot == 0:
        return 1.0 if ss_res == 0 else 0.0

    return float(1 - ss_res / ss_tot)