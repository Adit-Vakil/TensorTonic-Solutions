import numpy as np

def classification_metrics(y_true: list[int], y_pred: list[int], average: str = "micro", pos_label: int = 1) -> dict:
    n = len(y_true)
    accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / n
    tp, fp, fn, support = {}, {}, {}, {}
    for t, p in zip(y_true, y_pred):
        support[t] = support.get(t, 0) + 1
        if t == p:
            tp[t] = tp.get(t, 0) + 1
        else:
            fp[p] = fp.get(p, 0) + 1
            fn[t] = fn.get(t, 0) + 1
    classes = set(y_true) | set(y_pred)

    def prf(c):
        tp_, fp_, fn_ = tp.get(c, 0), fp.get(c, 0), fn.get(c, 0)
        p = tp_ / (tp_ + fp_) if (tp_ + fp_) else 0.0
        r = tp_ / (tp_ + fn_) if (tp_ + fn_) else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    if average == "micro":
        TP, FP, FN = sum(tp.values()), sum(fp.values()), sum(fn.values())
        precision = TP / (TP + FP) if (TP + FP) else 0.0
        recall = TP / (TP + FN) if (TP + FN) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    elif average == "macro":
        vals = [prf(c) for c in classes]
        precision = sum(v[0] for v in vals) / len(vals)
        recall = sum(v[1] for v in vals) / len(vals)
        f1 = sum(v[2] for v in vals) / len(vals)
    elif average == "weighted":
        total = sum(support.values())
        precision = recall = f1 = 0.0
        for c in classes:
            p, r, f = prf(c)
            w = support.get(c, 0) / total if total else 0.0
            precision += p * w; recall += r * w; f1 += f * w
    elif average == "binary":
        precision, recall, f1 = prf(pos_label)
    else:
        raise ValueError("average must be one of: micro, macro, weighted, binary")

    return {"accuracy": round(accuracy, 6), "precision": round(precision, 6),
            "recall": round(recall, 6), "f1": round(f1, 6)}