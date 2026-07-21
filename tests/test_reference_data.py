"""Validation of the refactored generators against the shipped reference data.

These tests regenerate small portions of the published tables from scratch and
compare them, cell for cell, with the CSVs in the repository's ``data``
directory.  They are marked ``slow`` because the two-dimensional regeneration
sweeps roughly half a million pairs.
"""

from __future__ import annotations

import pytest

from euclid_analysis import dataio
from euclid_analysis.frequencies import euclid_alg_frequencies, heilbronn


@pytest.mark.slow
class TestRegenerationMatchesReference:
    def test_two_dimensional_column_1001(self, dataset_C, dataset_B):
        _, B, C = euclid_alg_frequencies([1], [], [1, 1001], {}, {}, {})
        assert C[1001] == dataset_C[1001]
        assert B[1001] == dataset_B[1001]

    def test_one_dimensional_columns(self, dataset_H, dataset_H1):
        H1, H = heilbronn([1], list(range(1, 1001)))
        assert H[1000] == dataset_H[1000]
        assert H1[1000] == dataset_H1[1000]
        # Spot-check a couple of interior columns too.
        assert H[137] == dataset_H[137]
        assert H1[999] == dataset_H1[999]


@pytest.mark.slow
class TestFullRoundTrip:
    @pytest.mark.parametrize("name", ["B", "C", "H", "H1"])
    def test_dataset_round_trips_through_csv(self, name, tmp_path):
        path = dataio.DATA_DIR / dataio.DATASETS[name]
        if not path.exists():
            pytest.skip(f"dataset {name!r} not present")
        original = dataio.load_dataset(name)
        out = tmp_path / f"{name}.csv"
        dataio.save_meta_dict(original, out)
        assert dataio.load_meta_dict(out) == original
