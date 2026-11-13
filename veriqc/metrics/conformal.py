"""Split-conformal calibration of the triage thresholds.

The station's promise is a *controllable* error budget: for a target false
alarm rate ``alpha`` on normals, the threshold that realises it is not a tuned
constant but an order statistic of the calibration set, with a finite-sample
guarantee attached.

Split-conformal (Vovk et al.; Lei & Wasserman): with ``n`` calibration scores
``s_1 <= ... <= s_n`` from exchangeable normals and a target rate ``alpha``, the
threshold

    t = s_(k),   k = ceil((n+1)(1-alpha))

satisfies ``P(s_{n+1} > t) <= alpha`` for a new exchangeable normal, as long as
``k <= n`` (i.e. ``alpha >= 1/(n+1)``).  The threshold is the *k-th smallest
order statistic*; interpolating quantiles would break the guarantee, so the
order statistic is taken directly.

Two finite-sample statistical guarantees are provided:

1. **Split-Conformal Calibration**: Distribution-free threshold setting satisfying
   P(s_{n+1} > t) <= alpha for new exchangeable normals, valid when alpha >= 1 / (n + 1).
2. **Clopper-Pearson Exact Confidence Bound**: The 95% upper bound on observed false
   alarm rate over n calibration normals. For instance, certifying a 1% FPR requires
   at least 299 normal samples with zero observed exceedances.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import beta

__all__ = [
    "ConformalThreshold",
    "conformal_threshold",
    "cp_upper_fpr",
    "cp_lower_fpr",
    "min_n_for_target",
    "calibrate_thresholds",
]


@dataclass
class ConformalThreshold:
    """A calibrated threshold plus the bookkeeping needed to audit it."""

    alpha: float
    threshold: float
    n_calibration: int
    order_statistic_k: int
    achievable: bool
    cp_upper_fpr_zero_exceed: float
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def conformal_threshold(calibration_scores: np.ndarray, alpha: float) -> ConformalThreshold:
    """Split-conformal threshold at target rate ``alpha``.

    Returns a record with ``achievable=False`` when ``n`` is too small to
    support the requested rate, rather than silently returning a threshold
    that has no guarantee.
    """
    s = np.sort(np.asarray(calibration_scores, dtype=np.float64).reshape(-1))
    n = len(s)
    if n == 0:
        raise ValueError("no calibration scores")
    k = math.ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return ConformalThreshold(
            alpha=alpha,
            threshold=float("inf"),
            n_calibration=n,
            order_statistic_k=k,
            achievable=False,
            cp_upper_fpr_zero_exceed=cp_upper_fpr(n, 0),
            note=(
                f"alpha={alpha} needs n>={math.ceil(1 / alpha) - 1} calibration normals, "
                f"have {n}; smallest supportable alpha is {1.0 / (n + 1):.4f}"
            ),
        )
    # k-th smallest order statistic (1-based) -- no interpolation
    thr = float(s[k - 1])
    return ConformalThreshold(
        alpha=alpha,
        threshold=thr,
        n_calibration=n,
        order_statistic_k=k,
        achievable=True,
        cp_upper_fpr_zero_exceed=cp_upper_fpr(n, 0),
        note=f"k=ceil((n+1)(1-alpha))={k}, threshold=s_({k}) of n={n}",
    )


def cp_upper_fpr(n: int, k: int, confidence: float = 0.95) -> float:
    """Clopper-Pearson upper confidence bound on a rate from ``k`` successes in ``n``.

    With ``k=0`` this is ``1 - alpha_q^(1/n)``, the bound quoted by
    arXiv 2608.15090.
    """
    if n <= 0:
        return float("nan")
    alpha_q = 1.0 - confidence
    if k >= n:
        return 1.0
    return float(beta.ppf(1.0 - alpha_q, k + 1, n - k))


def cp_lower_fpr(n: int, k: int, confidence: float = 0.95) -> float:
    """Clopper-Pearson lower confidence bound on a rate from ``k`` successes in ``n``."""
    if n <= 0:
        return float("nan")
    alpha_q = 1.0 - confidence
    if k <= 0:
        return 0.0
    return float(beta.ppf(alpha_q, k, n - k + 1))


def min_n_for_target(alpha: float, confidence: float = 0.95) -> int:
    """Smallest ``n`` of normals that supports an FPR target of ``alpha``.

    Uses the conservative reading of arXiv 2608.15090: zero observed false
    alarms in ``n`` normals, 95 % upper bound on the FPR must not exceed
    ``alpha``.  Gives 299 for a 1 % target and 59 for a 5 % target.
    """
    alpha_q = 1.0 - confidence
    if alpha <= 0 or alpha >= 1:
        raise ValueError("alpha must be in (0, 1)")
    n = math.ceil(math.log(alpha_q) / math.log(1.0 - alpha))
    return int(n)


def calibrate_thresholds(
    calibration_scores: np.ndarray,
    alphas: tuple[float, ...] = (0.05, 0.01),
) -> dict[float, ConformalThreshold]:
    """Calibrate one threshold per target rate."""
    return {a: conformal_threshold(calibration_scores, a) for a in alphas}
