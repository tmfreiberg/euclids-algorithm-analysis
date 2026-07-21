"""Shared pytest fixtures and configuration.

Selects a non-interactive Matplotlib backend so plotting tests run headlessly,
and exposes the shipped reference datasets as session-scoped fixtures.  Any
dataset whose file is absent (notably the large ``A`` gcd-pair table) causes the
dependent fixture to skip rather than fail.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")

from euclid_analysis.dataio import DATA_DIR, DATASETS, load_dataset  # noqa: E402


def _load_or_skip(name: str):
    path = DATA_DIR / DATASETS[name]
    if not path.exists():
        pytest.skip(f"reference dataset {name!r} not present at {path}")
    return load_dataset(name)


@pytest.fixture(scope="session")
def dataset_C():
    """Two-dimensional step-count table over all pairs (dataset ``C``)."""
    return _load_or_skip("C")


@pytest.fixture(scope="session")
def dataset_B():
    """Two-dimensional step-count table over coprime pairs (dataset ``B``)."""
    return _load_or_skip("B")


@pytest.fixture(scope="session")
def dataset_H():
    """One-dimensional step-count table over all pairs (dataset ``H``)."""
    return _load_or_skip("H")


@pytest.fixture(scope="session")
def dataset_H1():
    """One-dimensional step-count table over coprime pairs (dataset ``H1``)."""
    return _load_or_skip("H1")
