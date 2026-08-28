from collections import Counter
import numpy as np

def mean_median_mode(x: list) -> dict:
    x = np.array(x, dtype = float)
    x_mn = float(np.mean(x))
    x_md = float(np.median(x))
    counts = Counter(x)
    x_mo = float(counts.most_common(1)[0][0])
    return {"mean": x_mn, "median": x_md,"mode": x_mo}