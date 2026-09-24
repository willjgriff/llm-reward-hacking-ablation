"""Stage 4 analysis: difference-of-means directions, held-out probes, projections and direction comparisons.

All functions take float32 numpy arrays of shape (n, hidden) for one layer. Following heretic: directions
are the raw difference of class means (float32), optionally after symmetric winsorization of the
activations; "unit" versions are normalised per layer.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler


def winsorize(x: np.ndarray, quantile: float | None) -> np.ndarray:
    if quantile is None or quantile >= 1.0:
        return x
    limit = np.quantile(np.abs(x), quantile)
    return np.clip(x, -limit, limit)


def diff_of_means(x_pos: np.ndarray, x_neg: np.ndarray, winsor_q: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (direction, mean_pos, mean_neg); direction = mean_pos - mean_neg (not normalised)."""
    x_pos = winsorize(x_pos.astype(np.float32), winsor_q)
    x_neg = winsorize(x_neg.astype(np.float32), winsor_q)
    mp, mn = x_pos.mean(axis=0), x_neg.mean(axis=0)
    return mp - mn, mp, mn


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def projection_scores(direction: np.ndarray, mean_pos: np.ndarray, mean_neg: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Scalar score per row along the unit direction, normalised so the class means map to 0 (neg) and 1 (pos)."""
    u = unit(direction)
    span = float(np.dot(mean_pos - mean_neg, u))
    if span == 0:
        return np.zeros(len(x), dtype=np.float32)
    return ((x - mean_neg) @ u) / span


def auroc(scores: np.ndarray, y: np.ndarray) -> float:
    y = np.asarray(y)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, scores))


def projection_auc(direction: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    return auroc(x @ unit(direction), y)


def fit_probe(x_train: np.ndarray, y_train: np.ndarray, C: float = 1.0, max_iter: int = 500, seed: int = 0):
    scaler = StandardScaler().fit(x_train)
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=max_iter, random_state=seed)
    clf.fit(scaler.transform(x_train), y_train)
    return scaler, clf


def eval_probe(probe, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    scaler, clf = probe
    z = scaler.transform(x)
    p = clf.decision_function(z)
    return {"auc": auroc(p, y), "balanced_accuracy": float(balanced_accuracy_score(y, (p > 0).astype(int)))}


def split_half_cosine(x_pos: np.ndarray, g_pos: np.ndarray, x_neg: np.ndarray, g_neg: np.ndarray, reps: int, seed: int, winsor_q=None) -> np.ndarray:
    """Cosine between directions from two disjoint halves of the (task-)groups, `reps` random halvings."""
    rng = np.random.default_rng(seed)
    groups = np.array(sorted(set(g_pos) | set(g_neg)))
    out = []
    for _ in range(reps):
        perm = rng.permutation(groups)
        half = set(perm[: len(perm) // 2])
        a_pos, a_neg = np.array([g in half for g in g_pos]), np.array([g in half for g in g_neg])
        if a_pos.sum() < 2 or (~a_pos).sum() < 2 or a_neg.sum() < 2 or (~a_neg).sum() < 2:
            out.append(float("nan"))
            continue
        d1, _, _ = diff_of_means(x_pos[a_pos], x_neg[a_neg], winsor_q)
        d2, _, _ = diff_of_means(x_pos[~a_pos], x_neg[~a_neg], winsor_q)
        out.append(cosine(d1, d2))
    return np.array(out)


def sample_efficiency(
    x_pos: np.ndarray, x_neg: np.ndarray, full_direction: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray,
    n: int, reps: int, seed: int, winsor_q=None,
) -> dict[str, float]:
    """Directions from `n` random positives and `n` random negatives: cosine to the full direction and eval AUROC."""
    rng = np.random.default_rng(seed)
    cos, aucs = [], []
    for _ in range(reps):
        ip = rng.choice(len(x_pos), size=min(n, len(x_pos)), replace=False)
        ineg = rng.choice(len(x_neg), size=min(n, len(x_neg)), replace=False)
        d, _, _ = diff_of_means(x_pos[ip], x_neg[ineg], winsor_q)
        cos.append(cosine(d, full_direction))
        aucs.append(projection_auc(d, x_eval, y_eval))
    return {"cos_mean": float(np.nanmean(cos)), "cos_std": float(np.nanstd(cos)), "auc_mean": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs))}
