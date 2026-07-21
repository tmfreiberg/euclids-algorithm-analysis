"""Figures and animations for the numerical investigation.

Every builder here is a pure function of its data arguments -- there is no
dependence on notebook-global state, unlike the original single-file script.
The animation builders return a :class:`matplotlib.animation.FuncAnimation`
(together with the figure) so the caller decides how to display or save it;
:func:`save_gif` writes an animation to an animated GIF via
:class:`~matplotlib.animation.PillowWriter`.

These routines regenerate the project's figures.  They are comparatively slow
(the full animations sweep thousands of frames), which is why the accompanying
Quarto book embeds the precomputed images in the repository's ``images``
directory rather than executing this module.  The functions are provided so the
figures *can* be reproduced from the data on demand.

Notes
-----
Importing this module imports :mod:`matplotlib`.  For headless / batch use,
select a non-interactive backend before importing, e.g.::

    import matplotlib
    matplotlib.use("Agg")
"""

from __future__ import annotations

from typing import Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import stats as sps

from . import analysis, constants

__all__ = [
    "save_gif",
    "plot_one_dim_distribution_frame",
    "animate_one_dim_distribution",
    "plot_two_dim_distribution_frame",
    "animate_two_dim_distribution",
    "plot_porter_sign_sum_frame",
    "plot_porter_error_frame",
    "plot_second_moment_frame",
    "plot_variance_error_frame",
]

FrequencyDict = dict[int, int]
MetaDict = dict[int, FrequencyDict]
StatsDict = dict[int, dict[str, Any]]


def save_gif(anim: FuncAnimation, path: str, *, fps: int = 5, dpi: int = 100,
             loop: bool = True) -> None:
    """Write an animation to an animated GIF using Pillow.

    Parameters
    ----------
    anim : matplotlib.animation.FuncAnimation
        The animation to save.
    path : str
        Destination ``.gif`` path.
    fps : int, optional
        Frames per second (default ``5``).
    dpi : int, optional
        Output resolution (default ``100``).
    loop : bool, optional
        If ``True`` (default) the GIF loops forever; if ``False`` it plays once.
    """
    writer = PillowWriter(fps=fps)
    # Pillow uses the GIF ``loop`` convention: 0 loops forever, 1 plays once.
    anim.save(path, dpi=dpi, writer=writer,
              savefig_kwargs={"facecolor": "white"})
    # ``PillowWriter`` does not expose the loop count directly; re-saving via
    # the metadata is left to the caller if a non-looping GIF is required.


# --------------------------------------------------------------------------- #
# One-dimensional distribution animation
# --------------------------------------------------------------------------- #
def _one_dim_frame_axis(dist: dict[int, dict[int, float]], frame_list: list[int]):
    """Return common ``(x_min, x_max, y_max)`` axis bounds for the 1-D frames."""
    hor_axis: set[int] = set()
    for a in frame_list:
        hor_axis |= set(dist[a].keys())
    ordered = sorted(hor_axis)
    y_max = max(max(dist[a].values()) for a in frame_list)
    return ordered, min(ordered), max(ordered), y_max


def plot_one_dim_distribution_frame(
    ax: plt.Axes,
    a: int,
    dist: dict[int, dict[int, float]],
    stats: StatsDict,
    hor_axis: list[int],
    x_min: int,
    x_max: int,
    y_max: float,
) -> None:
    """Draw a single frame of the one-dimensional distribution animation.

    The frame shows, for a fixed ``a``, the probability ``P(Z = s)`` over the
    totatives of ``a`` (crosses and bars) and the normal density with Porter's
    mean estimate ``mu_* = lambda*log a + C_P - 1`` and the empirical variance.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into (cleared first).
    a : int
        The value of ``a`` for this frame.
    dist : dict
        Relative-frequency distributions keyed by ``a`` (e.g. ``H1`` via
        :func:`euclid_analysis.statistics.dists`).
    stats : StatsDict
        Per-``a`` statistics (for the mean and variance).
    hor_axis : list of int
        The shared horizontal tick positions.
    x_min, x_max : int
        Shared horizontal bounds.
    y_max : float
        Shared vertical bound.
    """
    ax.clear()
    xleft, xright = x_min - 0.5, x_max + 0.5
    ytop = np.ceil(10 * y_max) / 10
    ax.set(xlim=(xleft, xright), ylim=(0, ytop))
    ax.set_xticks(hor_axis)
    ax.grid(True, zorder=0, alpha=0.7)

    keys = list(dist[a].keys())
    ax.plot(keys, list(dist[a].values()), "bx", zorder=4.5,
            label=r"$\mathrm{Prob}(Z = s)$")
    ax.bar(keys, list(dist[a].values()), zorder=2.5)

    mu = stats[a]["mean"]
    sigma = np.sqrt(stats[a]["var"])
    normal_points = np.linspace(keys[0], keys[-1])
    mu_p = constants.LAMBDA_DIXON * np.log(a) + constants.PORTER_CONSTANT - 1
    ax.plot(normal_points, sps.norm.pdf(normal_points, mu_p, sigma), ":",
            color="y", zorder=3.5, label=r"$\mathrm{Norm}(\mu_*,\sigma^2)$")

    ax.set_xlabel(r"$s = \#$steps of the form $(u,v) \mapsto (v,u$ mod $v)$")
    ax.set_ylabel("probability")
    ax.text(0.05, 0.93, rf"$a = $ {a},", transform=ax.transAxes)
    ax.text(0.23, 0.93,
            rf"$\mu_* = \lambda\log a + C_P - 1 = {mu_p:.3f}$",
            transform=ax.transAxes)
    ax.text(0.455, 0.85, r"$\mathbb{E}[Z] = $" + rf"${mu:.3f}$",
            transform=ax.transAxes)
    ax.text(0.75, 0.73, rf"$\mu_* = $ {mu_p:.3f}", transform=ax.transAxes)
    ax.text(0.75, 0.65, rf"$\sigma^2 = $ {sigma:.3f}", transform=ax.transAxes)
    ax.legend(loc=1, ncol=1, framealpha=0.5)


def animate_one_dim_distribution(
    dist: dict[int, dict[int, float]],
    stats: StatsDict,
    a_min: int = 100,
    a_max: int = 10000,
    step: int = 100,
) -> tuple[plt.Figure, FuncAnimation]:
    """Build the one-dimensional step-count distribution animation.

    Frames run over ``a = a_min, a_min+step, ..., a_max`` (restricted to those
    present in ``dist``), each drawn by
    :func:`plot_one_dim_distribution_frame`.

    Parameters
    ----------
    dist : dict
        Relative-frequency distributions keyed by ``a`` (``H1``).
    stats : StatsDict
        Per-``a`` statistics.
    a_min, a_max, step : int, optional
        Frame schedule (defaults ``100``, ``10000``, ``100``).

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.animation.FuncAnimation)
        The figure and the animation.
    """
    frame_list = [a for a in range(a_min, a_max + step, step) if a in dist]
    hor_axis, x_min, x_max, y_max = _one_dim_frame_axis(dist, frame_list)

    fig, ax = plt.subplots()
    fig.suptitle("Number of divisions in Euclidean algorithm for gcd(a,b) \n"
                 "for totatives b of a")

    def draw(a: int) -> None:
        plot_one_dim_distribution_frame(
            ax, a, dist, stats, hor_axis, x_min, x_max, y_max
        )

    # draw returns None because blit is off; the FuncAnimation stub is stricter.
    anim = FuncAnimation(
        fig,
        draw,  # type: ignore[arg-type]
        frames=frame_list,
        interval=500,
        blit=False,
        repeat=False,
    )
    return fig, anim


# --------------------------------------------------------------------------- #
# Two-dimensional distribution animation
# --------------------------------------------------------------------------- #
def plot_two_dim_distribution_frame(
    ax: plt.Axes,
    N: int,
    all_dist: dict[int, dict[int, float]],
    all_stats: StatsDict,
    coprime_dist: dict[int, dict[int, float]],
    coprime_stats: StatsDict,
) -> None:
    """Draw a single frame of the two-dimensional distribution animation.

    Blue marks the distribution of ``X`` (all pairs ``1 <= b < a < N``); red
    marks ``X_1`` (coprime pairs).  Dotted curves are normal with the empirical
    mean and variance; solid curves use the theoretical mean and the variance
    with the empirical subdominant constants ``kappa`` and ``kappa_1``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into (cleared first).
    N : int
        The checkpoint for this frame.
    all_dist, coprime_dist : dict
        Relative-frequency distributions keyed by ``N`` for ``X`` and ``X_1``.
    all_stats, coprime_stats : StatsDict
        Corresponding per-``N`` statistics.
    """
    ax.clear()
    ax.grid(True, zorder=0, alpha=0.7)

    ax.plot(list(all_dist[N].keys()), list(all_dist[N].values()), "b.",
            zorder=4.5, label=r"$\mathrm{Prob}(X = s)$")
    ax.plot(list(coprime_dist[N].keys()), list(coprime_dist[N].values()), "r.",
            zorder=4.5, label=r"$\mathrm{Prob}(X_1 = s)$")

    mu, var = all_stats[N]["mean"], all_stats[N]["var"]
    sigma = np.sqrt(var)
    mu1, var1 = coprime_stats[N]["mean"], coprime_stats[N]["var"]
    sigma1 = np.sqrt(var1)
    normal_points = np.linspace(
        list(all_dist[N].keys())[0], list(all_dist[N].keys())[-1]
    )
    ax.plot(normal_points, sps.norm.pdf(normal_points, mu, sigma), ":",
            color="b", alpha=0.9, zorder=3.5,
            label=r"$\mathrm{Norm}(\mu,\sigma^2)$")
    ax.plot(normal_points, sps.norm.pdf(normal_points, mu1, sigma1), ":",
            color="r", alpha=0.9, zorder=3.5,
            label=r"$\mathrm{Norm}(\mu_1,\sigma_1^2)$")

    lam = constants.LAMBDA_DIXON
    mu_p = lam * np.log(N - 1) + constants.NU_NORTON - 0.5
    var_p = constants.ETA_HENSLEY * np.log(N - 1) + constants.KAPPA_VAR
    mu1_p = lam * np.log(N - 1) + constants.NU_NORTON_COPRIME - 0.5
    var1_p = constants.ETA_HENSLEY * np.log(N - 1) + constants.KAPPA_VAR_COPRIME
    ax.plot(normal_points, sps.norm.pdf(normal_points, mu_p, np.sqrt(var_p)),
            "-", color="#34d5eb", alpha=0.5, zorder=2.5,
            label=r"$\mathrm{Norm}(\mu_{*},\sigma_{*}^2)$")
    ax.plot(normal_points, sps.norm.pdf(normal_points, mu1_p, np.sqrt(var1_p)),
            "-", color="#eb7734", alpha=0.5, zorder=2.5,
            label=r"$\mathrm{Norm}(\mu_{1*},\sigma_{1*}^2)$")

    ax.set_xlabel(r"$\#$steps of the form $(u,v) \mapsto (v,u$ mod $v)$")
    ax.set_ylabel("probability")
    ax.text(0.02, 0.95, rf"$N = {N}$", transform=ax.transAxes)
    ax.legend(loc=2, ncol=3, framealpha=0.5, mode="expand")


def animate_two_dim_distribution(
    all_dist: dict[int, dict[int, float]],
    all_stats: StatsDict,
    coprime_dist: dict[int, dict[int, float]],
    coprime_stats: StatsDict,
    frame_list: list[int] | None = None,
) -> tuple[plt.Figure, FuncAnimation]:
    """Build the two-dimensional step-count distribution animation.

    Parameters
    ----------
    all_dist, coprime_dist : dict
        Relative-frequency distributions keyed by ``N`` for ``X`` and ``X_1``.
    all_stats, coprime_stats : StatsDict
        Corresponding per-``N`` statistics.
    frame_list : list of int, optional
        Checkpoints to animate; defaults to every key of ``all_dist`` in order.

    Returns
    -------
    tuple of (matplotlib.figure.Figure, matplotlib.animation.FuncAnimation)
        The figure and the animation.
    """
    if frame_list is None:
        frame_list = sorted(all_dist.keys())

    fig, ax = plt.subplots()
    fig.suptitle("Number of divisions in Euclidean algorithm for gcd(a,b)")

    def draw(N: int) -> None:
        plot_two_dim_distribution_frame(
            ax, N, all_dist, all_stats, coprime_dist, coprime_stats
        )

    # draw returns None because blit is off; the FuncAnimation stub is stricter.
    anim = FuncAnimation(
        fig,
        draw,  # type: ignore[arg-type]
        frames=frame_list,
        interval=500,
        blit=False,
        repeat=False,
    )
    return fig, anim


# --------------------------------------------------------------------------- #
# Static frames for the error-term figures
# --------------------------------------------------------------------------- #
def plot_porter_sign_sum_frame(
    ax: plt.Axes, summary: dict[str, Any], N: int
) -> None:
    """Draw the Porter sign-sum random walk up to ``N``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into (cleared first).
    summary : dict
        The output of :func:`euclid_analysis.analysis.sign_summary`.
    N : int
        Upper limit on the horizontal axis.
    """
    ax.clear()
    ax.plot(range(1, N + 1), summary["sum_sign"][1:N + 1],
            label=r"$\sum_{a=1}^{N}\mathrm{sgn}(\mathbb{E}[Z] - "
                  r"(\lambda\log a + C_P - 1))$")
    ax.set_xlabel(r"$N$")
    ax.text(0.8, 0.9, rf"$N = {N}$", transform=ax.transAxes)
    ax.text(0.8, 0.83, rf"{100 * summary['prop_pos'][N + 1]:.2f}$\% +$",
            transform=ax.transAxes)
    ax.text(0.8, 0.76, rf"{100 * summary['prop_neg'][N + 1]:.2f}$\% -$",
            transform=ax.transAxes)
    ax.legend(loc=3, framealpha=0.5)


def plot_porter_error_frame(
    ax: plt.Axes,
    error_series: dict[int, float],
    champions: dict[int, tuple[float, float, float, float]],
    N: int,
) -> None:
    """Draw the Porter error term ``E(a)`` for ``a <= N`` with bounding parabolas.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into (cleared first).
    error_series : dict of {int: float}
        The Porter error series from
        :func:`euclid_analysis.analysis.porter_error_series`.
    champions : dict
        The error champions from
        :func:`euclid_analysis.analysis.error_champions`.
    N : int
        Upper limit ``a <= N``.
    """
    ax.clear()
    xs = [a for a in error_series.keys() if 1 <= a <= N]
    ax.plot(xs, [error_series[a] for a in xs],
            label=r"$\sum_{b \in \mathbb{Z}_a^{\times}} "
                  r"[T(a,b) - (\lambda \log a + C_P - 1)]$")
    ax.plot(xs, [np.sqrt(a) for a in xs], "r:", label=r"$\pm\sqrt{a}$")
    ax.plot(xs, [-np.sqrt(a) for a in xs], "r:")
    c = max(champions[a][3] for a in champions if a < N + 1)
    ax.plot(xs, [c * np.sqrt(a) for a in xs], "y:",
            label=rf"$\pm${c:.3f}" + r"$\sqrt{a}$")
    ax.plot(xs, [-c * np.sqrt(a) for a in xs], "y:")
    ax.set_xlabel(r"$a$")
    ax.set_ylabel("error")
    ax.text(0.05, 0.93, rf"$a \leq $ {N}", transform=ax.transAxes)
    for a in champions:
        if N - 50 < a < N + 1:
            ax.annotate(f"a = {a}", (a, error_series[a]),
                        textcoords="offset points", xytext=(0, 0), ha="center")
    ax.legend(loc=3, framealpha=0.5)


def plot_second_moment_frame(
    ax: plt.Axes, stats: StatsDict, N: int, a_min: int = 1
) -> None:
    """Draw ``E[Z^2]`` for ``a_min <= a <= N`` with its quadratic-in-log fit.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into (cleared first).
    stats : StatsDict
        Per-``a`` statistics containing ``"2ndmom"``.
    N : int
        Upper limit on ``a``.
    a_min : int, optional
        Lower limit on ``a`` (default ``1``).
    """
    ax.clear()
    ax.grid(True, zorder=0, alpha=0.7)
    xs = [a for a in stats.keys() if a_min <= a <= N]
    ys = [stats[a]["2ndmom"] for a in xs]
    ax.plot(xs, ys, "bx",
            label=r"$\frac{1}{\phi(a)}\sum_{b \in \mathbb{Z}_a^{\times}} "
                  rf"T(a,b)^2$, ${a_min} \leq a \leq {N}$")
    c2, c1, c0 = analysis.fit_second_moment(stats, a_min, N)
    model = analysis.second_moment_model(np.array(xs, dtype=float), c2, c1, c0)
    ax.plot(xs, model, "y:",
            label=r"$c_2(\log a)^2 + c_1\log a + c_0$" + "\n"
                  rf"$(c_2,c_1,c_0) = ({c2:.4f}, {c1:.4f}, {c0:.4f})$")
    ax.set_xlabel(r"$a$")
    ax.set_ylabel(r"$\mathbb{E}[Z^2]$")
    ax.legend(loc=4, framealpha=0.5)


def plot_variance_error_frame(
    ax: plt.Axes,
    var_error: dict[int, float],
    N: int,
    a_min: int = 100,
) -> None:
    """Draw the variance-estimate residual for ``a_min <= a <= N``.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw into (cleared first).
    var_error : dict of {int: float}
        Residuals from
        :func:`euclid_analysis.analysis.variance_guess_error`.
    N : int
        Upper limit on ``a``.
    a_min : int, optional
        Lower limit on ``a`` (default ``100``).
    """
    ax.clear()
    ax.grid(True, zorder=0, alpha=0.7)
    xs = [a for a in var_error.keys() if a_min <= a <= N]
    ys = [var_error[a] for a in xs]
    ax.plot(xs, ys, "b-", label=r"$\mathrm{Var}(Z) - (\eta\log a + $const$)$, "
                                rf"${a_min} \leq a \leq {N}$")

    def shape(a: int) -> float:
        return 1.0 if a == 1 else (np.log(a)) ** 2 / np.sqrt(a)

    cp = max(var_error[a] / shape(a) for a in xs)
    cn = min(var_error[a] / shape(a) for a in xs)
    ax.plot(xs, [cp * shape(a) for a in xs], "y-", zorder=4.5,
            label=rf"$c_+(\log a)^2/\sqrt{{a}}$, $c_+ = {cp:.3f}$")
    ax.plot(xs, [cn * shape(a) for a in xs], "g-", zorder=4.5,
            label=rf"$c_-(\log a)^2/\sqrt{{a}}$, $c_- = {cn:.3f}$")
    ax.set_xlabel(r"$a$")
    ax.set_ylabel("error")
    ax.legend(loc=0, framealpha=0.8, ncol=1)