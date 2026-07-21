"""Headless smoke tests for :mod:`euclid_analysis.plotting`.

These verify that each figure builder runs and that the animation-to-GIF path
works; they do not assert on pixels.  The Matplotlib backend is set to ``Agg``
in :mod:`tests.conftest`.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from euclid_analysis import analysis as an
from euclid_analysis import plotting as pl
from euclid_analysis.statistics import basic_stats, dists


def test_one_dim_frame_renders(dataset_H1):
    stats = basic_stats(dataset_H1)
    dist = dists(dataset_H1)
    frames = [a for a in range(100, 1001, 100) if a in dist]
    hor = sorted({s for a in frames for s in dist[a]})
    y_max = max(max(dist[a].values()) for a in frames)
    fig, ax = plt.subplots()
    pl.plot_one_dim_distribution_frame(
        ax, 1000, dist, stats, hor, min(hor), max(hor), y_max
    )
    plt.close(fig)


def test_two_dim_frame_renders(dataset_C, dataset_B):
    fig, ax = plt.subplots()
    pl.plot_two_dim_distribution_frame(
        ax, 1001, dists(dataset_C), basic_stats(dataset_C),
        dists(dataset_B), basic_stats(dataset_B),
    )
    plt.close(fig)


def test_porter_frames_render(dataset_H1):
    series = an.porter_error_series(dataset_H1)
    champs = an.error_champions(series)
    summ = an.sign_summary(series)
    fig, ax = plt.subplots()
    pl.plot_porter_sign_sum_frame(ax, summ, 500)
    pl.plot_porter_error_frame(ax, series, champs, 500)
    pl.plot_second_moment_frame(ax, basic_stats(dataset_H1), 500)
    pl.plot_variance_error_frame(
        ax, an.variance_guess_error(basic_stats(dataset_H1)), 500
    )
    plt.close(fig)


def test_animation_saves_gif(dataset_C, dataset_B, tmp_path):
    fig, anim = pl.animate_two_dim_distribution(
        dists(dataset_C), basic_stats(dataset_C),
        dists(dataset_B), basic_stats(dataset_B),
        frame_list=[1001, 2001, 3001],
    )
    out = tmp_path / "anim.gif"
    pl.save_gif(anim, str(out), fps=2)
    assert out.exists() and out.stat().st_size > 0
    plt.close(fig)
