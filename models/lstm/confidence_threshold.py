import numpy as np


def eval_with_ignore_zone(y_true: np.ndarray, p: np.ndarray, threshold: float):
    y_true = y_true.reshape(-1)
    p = p.reshape(-1)

    TP = FP = TN = FN = ignored = 0

    for yt, prob in zip(y_true, p):
        if prob >= threshold:
            pred = 1
        elif prob < 1.0 - threshold:
            pred = 0
        else:
            ignored += 1
            continue

        if yt == 1 and pred == 1:
            TP += 1
        elif yt == 0 and pred == 1:
            FP += 1
        elif yt == 0 and pred == 0:
            TN += 1
        elif yt == 1 and pred == 0:
            FN += 1

    total = TP + FP + TN + FN
    eps = 1e-8
    acc = (TP + TN) / (total + eps)
    prec = TP / (TP + FP + eps)
    rec = TP / (TP + FN + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    coverage = total / (total + ignored + eps)

    return {
        "threshold": threshold,
        "coverage": float(coverage),
        "ignored": int(ignored),
        "used_samples": int(total),
        "TP": int(TP), "FP": int(FP), "TN": int(TN), "FN": int(FN),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
    }