"""Numerical investigation of the mean and variance error terms.

This module contains the *fast, data-driven* numerics behind the exposition's
figures: the quantities that are computed from an already-tabulated frequency
table and compared against the theoretical constants.  Keeping them here (apart
from the plotting code) makes them importable and testable in isolation.

One-dimensional error term (Porter)
-----------------------------------
For a fixed ``a`` and the random variable ``Z`` giving ``T(a, b)`` over
totatives ``b`` of ``a``, Porter's theorem estimates ``E[Z]`` by
``lambda*log(a) + C_P - 1``.  The signed error, summed over the totatives, is

    E(a) = sum_{b in Z_a^x} [ T(a, b) - (lambda*log(a) + C_P - 1) ].

:func:`porter_error` computes ``E(a)`` from a coprime frequency table, and
:func:`porter_error_series` produces the whole ``{a: E(a)}`` mapping.  The sign
of ``E(a)`` is expected to behave like a fair coin (:func:`sign_summary`), and
the record-setting values of ``E(a)/sqrt(a)`` are the "error champions"
(:func:`error_champions`).

Two-dimensional error terms (Norton, Hensley, Baladi--Vallée)
-------------------------------------------------------------
For ``X`` (all pairs) and ``X_1`` (coprime pairs) over ``1 <= b < a <= N``,
:func:`mean_variance_error_terms` assembles the six series
``E[X]-lambda*log N``, ``E[X_1]-lambda*log N``, ``Var(X)-eta*log N``,
``Var(X_1)-eta*log N`` and their coprime/all differences, and
:func:`fit_subdominant_constants` curve-fits each to ``c/sqrt(N) + const`` to
estimate the subdominant constants ``B``, ``B_1``, ``D``, ``D_1`` -- the last
two being empirical estimates of ``kappa`` and ``kappa_1``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import curve_fit

from . import constants as C

__all__ = [
    "porter_error",
    "porter_error_series",
    "sign_summary",
    "error_champions",
    "second_moment_model",
    "fit_second_moment",
    "variance_guess_error",
    "mean_variance_error_terms",
    "inverse_sqrt_model",
    "fit_subdominant_constants",
]

FrequencyDict = dict[int, int]
MetaDict = dict[int, FrequencyDict]
StatsDict = dict[int, dict[str, Any]]


# --------------------------------------------------------------------------- #
# One-dimensional: Porter error term
# --------------------------------------------------------------------------- #
def porter_error(frequencies: FrequencyDict, a: int) -> float:
    """Return the summed Porter error ``E(a)`` for a single ``a``.

    Computes ``sum_s (s - (lambda*log(a) + C_P - 1)) * frequencies[s]`` where
    ``frequencies`` is the step-count distribution over the totatives of ``a``
    (i.e. one column of the coprime one-dimensional table ``H1``).

    Parameters
    ----------
    frequencies : dict of {int: int}
        Step-count frequencies ``{s: count}`` for the given ``a``.
    a : int
        The value of ``a`` (used in the logarithmic mean estimate).

    Returns
    -------
    float
        The signed error ``E(a)``.
    """
    mu_p = C.LAMBDA_DIXON * np.log(a) + C.PORTER_CONSTANT - 1
    return float(sum((s - mu_p) * frequencies[s] for s in frequencies.keys()))


def porter_error_series(coprime_meta: MetaDict) -> dict[int, float]:
    """Return ``{a: E(a)}`` for every ``a`` in a coprime frequency table.

    Parameters
    ----------
    coprime_meta : MetaDict
        The one-dimensional coprime table ``H1`` mapping each ``a`` to its
        step-count distribution over totatives.

    Returns
    -------
    dict of {int: float}
        The Porter error term for each ``a``.
    """
    return {a: porter_error(coprime_meta[a], a) for a in coprime_meta.keys()}


def sign_summary(error_series: dict[int, float]) -> dict[str, Any]:
    """Summarise the sign behaviour of a Porter-error series as a random walk.

    Interpreting ``sgn(E(a))`` as +/-1 steps, this accumulates the running
    counts of positive and negative values and their partial-sum walk.  It
    assumes ``error_series`` is keyed by every integer ``a`` from ``1`` upward.

    Parameters
    ----------
    error_series : dict of {int: float}
        A ``{a: E(a)}`` mapping, as from :func:`porter_error_series`.

    Returns
    -------
    dict
        With keys ``"pos"`` and ``"neg"`` (cumulative counts, length ``N+1``,
        index ``a`` giving the count up to ``a``), ``"prop_pos"`` /
        ``"prop_neg"`` (the corresponding proportions), and ``"sum_sign"``
        (the signed partial-sum walk ``pos[a] - neg[a]``, index ``a-1``).
    """
    pos = [0]
    neg = [0]
    for a in error_series.keys():
        if error_series[a] > 0:
            pos.append(pos[a - 1] + 1)
            neg.append(neg[a - 1])
        else:
            pos.append(pos[a - 1])
            neg.append(neg[a - 1] + 1)
    prop_pos = [0.0]
    prop_neg = [0.0]
    for a in error_series.keys():
        prop_pos.append(pos[a] / a)
        prop_neg.append(neg[a] / a)
    sum_sign = [pos[a] - neg[a] for a in error_series.keys()]
    return {
        "pos": pos,
        "neg": neg,
        "prop_pos": prop_pos,
        "prop_neg": prop_neg,
        "sum_sign": sum_sign,
    }


def error_champions(
    error_series: dict[int, float],
) -> dict[int, tuple[float, float, float, float]]:
    """Return the record-setting values of ``E(a)/sqrt(a)`` ("error champions").

    An ``a`` is a champion when ``E(a)/sqrt(a)`` exceeds every previous
    positive record or falls below every previous negative record.  These are
    the points at which the tightest bounding parabola ``+/- c*sqrt(a)``
    containing the error graph must widen.

    Parameters
    ----------
    error_series : dict of {int: float}
        A ``{a: E(a)}`` mapping.

    Returns
    -------
    dict
        ``{a: (E(a), c_pos, c_neg, c)}`` for each champion ``a``, where
        ``c_pos`` and ``c_neg`` are the running positive/negative records of
        ``E(a)/sqrt(a)`` and ``c = max(c_pos, -c_neg)``.
    """
    champions: dict[int, tuple[float, float, float, float]] = {}
    c_pos, c_neg = 0.0, 0.0
    for a in error_series.keys():
        ratio = error_series[a] / np.sqrt(a)
        if ratio > c_pos:
            c_pos = ratio
            champions[a] = (error_series[a], c_pos, c_neg, max(c_pos, -c_neg))
        if ratio < c_neg:
            c_neg = ratio
            champions[a] = (error_series[a], c_pos, c_neg, max(c_pos, -c_neg))
    return champions


# --------------------------------------------------------------------------- #
# One-dimensional: second moment and variance
# --------------------------------------------------------------------------- #
def second_moment_model(x: np.ndarray, c2: float, c1: float, c0: float) -> np.ndarray:
    """Quadratic-in-``log`` model ``c2*(log x)^2 + c1*log x + c0``.

    Parameters
    ----------
    x : numpy.ndarray
        Positive abscissae (values of ``a``).
    c2, c1, c0 : float
        Coefficients.

    Returns
    -------
    numpy.ndarray
        The model evaluated at ``x``.
    """
    return c2 * (np.log(x)) ** 2 + c1 * np.log(x) + c0


def fit_second_moment(
    stats: StatsDict, a_min: int = 1, a_max: int | None = None
) -> tuple[float, float, float]:
    """Curve-fit the second moment ``E[Z^2]`` to :func:`second_moment_model`.

    Parameters
    ----------
    stats : StatsDict
        A per-``a`` statistics mapping (as from
        :func:`euclid_analysis.statistics.basic_stats`) whose entries contain a
        ``"2ndmom"`` key.
    a_min, a_max : int, optional
        Inclusive range of ``a`` to fit over.  ``a_max=None`` (default) uses the
        largest available key.

    Returns
    -------
    tuple of (float, float, float)
        The fitted ``(c2, c1, c0)``.  Empirically ``c2`` is close to
        ``lambda**2``.
    """
    if a_max is None:
        a_max = max(stats.keys())
    xs = np.array([a for a in stats.keys() if a_min <= a <= a_max], dtype=float)
    ys = np.array([stats[int(a)]["2ndmom"] for a in xs], dtype=float)
    popt, _ = curve_fit(second_moment_model, xs, ys)
    return tuple(float(v) for v in popt)  # type: ignore[return-value]


def variance_guess_error(
    stats: StatsDict, constant: float = -0.354293955
) -> dict[int, float]:
    """Return ``Var(Z) - (eta*log a + constant)`` for each ``a``.

    Parameters
    ----------
    stats : StatsDict
        A per-``a`` statistics mapping whose entries contain a ``"var"`` key.
    constant : float, optional
        The subdominant constant subtracted from ``eta*log a`` (default is the
        empirically fitted value ``-0.354293955`` from the exposition).

    Returns
    -------
    dict of {int: float}
        The residual variance-estimate error for each ``a``.
    """
    return {
        a: stats[a]["var"] - (C.ETA_HENSLEY * np.log(a) + constant)
        for a in stats.keys()
    }


# --------------------------------------------------------------------------- #
# Two-dimensional: mean and variance error terms
# --------------------------------------------------------------------------- #
def mean_variance_error_terms(
    all_stats: StatsDict, coprime_stats: StatsDict
) -> dict[str, dict[int, float]]:
    """Assemble the two-dimensional mean/variance error-term series.

    For each checkpoint ``N`` present in both inputs this computes the six
    series used in the two-dimensional error-term figure, subtracting the known
    leading terms ``lambda*log(N-1)`` (mean) and ``eta*log(N-1)`` (variance).

    Parameters
    ----------
    all_stats : StatsDict
        Statistics for ``X`` over all pairs (dataset ``C``).
    coprime_stats : StatsDict
        Statistics for ``X_1`` over coprime pairs (dataset ``B``).

    Returns
    -------
    dict
        With keys ``"err_mean"``, ``"err_mean1"``, ``"err_var"``,
        ``"err_var1"``, ``"mean_delta"`` (``E[X_1]-E[X]``) and ``"var_delta"``
        (``Var(X_1)-Var(X)``), each a ``{N: value}`` mapping.
    """
    keys = [N for N in all_stats.keys() if N in coprime_stats]
    err_mean: dict[int, float] = {}
    err_mean1: dict[int, float] = {}
    err_var: dict[int, float] = {}
    err_var1: dict[int, float] = {}
    mean_delta: dict[int, float] = {}
    var_delta: dict[int, float] = {}
    for N in keys:
        log_n = np.log(N - 1)
        err_mean[N] = all_stats[N]["mean"] - C.LAMBDA_DIXON * log_n
        err_mean1[N] = coprime_stats[N]["mean"] - C.LAMBDA_DIXON * log_n
        err_var[N] = all_stats[N]["var"] - C.ETA_HENSLEY * log_n
        err_var1[N] = coprime_stats[N]["var"] - C.ETA_HENSLEY * log_n
        mean_delta[N] = coprime_stats[N]["mean"] - all_stats[N]["mean"]
        var_delta[N] = coprime_stats[N]["var"] - all_stats[N]["var"]
    return {
        "err_mean": err_mean,
        "err_mean1": err_mean1,
        "err_var": err_var,
        "err_var1": err_var1,
        "mean_delta": mean_delta,
        "var_delta": var_delta,
    }


def inverse_sqrt_model(x: np.ndarray, a: float, b: float) -> np.ndarray:
    """Square-root-cancellation model ``a/sqrt(x) + b``.

    Parameters
    ----------
    x : numpy.ndarray
        Positive abscissae (checkpoints ``N``).
    a, b : float
        Coefficients; ``b`` is the subdominant constant of interest.

    Returns
    -------
    numpy.ndarray
        The model evaluated at ``x``.
    """
    return a / np.sqrt(x) + b


def fit_subdominant_constants(
    error_series: dict[int, float],
    n_min: int | None = None,
    n_max: int | None = None,
) -> tuple[float, float]:
    """Fit an error series to ``c/sqrt(N) + const`` and return ``(c, const)``.

    Applied to the variance error series this estimates the subdominant
    constants ``kappa`` (dataset ``C``) and ``kappa_1`` (dataset ``B``); applied
    to the mean error series it estimates Norton's ``B`` and ``B_1``.

    Parameters
    ----------
    error_series : dict of {int: float}
        A ``{N: error}`` mapping, e.g. one of the series from
        :func:`mean_variance_error_terms`.
    n_min, n_max : int, optional
        Inclusive checkpoint range to fit over; ``None`` uses the extremes.

    Returns
    -------
    tuple of (float, float)
        The fitted ``(c, const)`` of ``c/sqrt(N) + const``.
    """
    ns = sorted(error_series.keys())
    if n_min is not None:
        ns = [n for n in ns if n >= n_min]
    if n_max is not None:
        ns = [n for n in ns if n <= n_max]
    xs = np.array(ns, dtype=float)
    ys = np.array([error_series[n] for n in ns], dtype=float)
    popt, _ = curve_fit(inverse_sqrt_model, xs, ys)
    return float(popt[0]), float(popt[1])
