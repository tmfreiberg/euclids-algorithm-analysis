"""Tests for :mod:`euclid_analysis.statistics`."""

from __future__ import annotations

import numpy as np
import pytest

from euclid_analysis import statistics as st


class TestDictionaryStatistics:
    def test_simple_distribution(self):
        s = st.dictionary_statistics({1: 2, 2: 2, 3: 1})
        assert s["mean"] == pytest.approx(1.8)
        assert s["2ndmom"] == pytest.approx((2 * 1 + 2 * 4 + 1 * 9) / 5)
        assert s["var"] == pytest.approx(s["2ndmom"] - s["mean"] ** 2)
        assert s["sdv"] == pytest.approx(np.sqrt(s["var"]))
        assert s["mode"] == [1, 2]
        assert set(s["dist"].keys()) == {1, 2, 3}
        assert s["dist"][1] == pytest.approx(0.4)

    def test_relative_frequencies_sum_to_one(self):
        s = st.dictionary_statistics({1: 3, 2: 5, 5: 2})
        assert sum(s["dist"].values()) == pytest.approx(1.0)

    def test_median_odd_count(self):
        # Data: 1,1,1,2,3 (n=5) -> median 1.
        s = st.dictionary_statistics({1: 3, 2: 1, 3: 1})
        assert s["med"] == 1

    def test_median_even_count(self):
        # Data: 1,1,2,2 (n=4) -> median 1.5.
        s = st.dictionary_statistics({1: 2, 2: 2})
        assert s["med"] == pytest.approx(1.5)

    def test_single_value(self):
        s = st.dictionary_statistics({4: 10})
        assert s["mean"] == 4
        assert s["var"] == pytest.approx(0.0)
        assert s["mode"] == [4]


class TestMetaHelpers:
    def test_basic_stats_and_dists_keys(self):
        meta = {10: {1: 2, 2: 3}, 20: {1: 1, 3: 4}}
        stats = st.basic_stats(meta)
        dists = st.dists(meta)
        assert set(stats.keys()) == {10, 20}
        assert dists[10] == stats[10]["dist"]

    def test_tabulate_integer_dtype_and_zero_fill(self):
        meta = {10: {1: 2, 2: 3}, 20: {1: 1, 3: 4}}
        frame = st.tabulate(meta)
        assert frame.loc[3, 10] == 0  # missing cell filled with zero
        assert frame.loc[1, 20] == 1
        assert str(frame.dtypes.iloc[0]).startswith("int")

    def test_tabulate_float_dtype_for_relative_frequencies(self):
        meta = {10: {1: 2, 2: 3}}
        frame = st.tabulate(st.dists(meta))
        assert str(frame.dtypes.iloc[0]).startswith("float")
