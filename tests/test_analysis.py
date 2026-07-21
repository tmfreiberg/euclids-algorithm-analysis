"""Tests for :mod:`euclid_analysis.analysis`.

The distribution-level checks use small synthetic inputs; the reproduction of
the published constants uses the shipped datasets and is skipped when they are
absent.
"""

from __future__ import annotations

import numpy as np
import pytest

from euclid_analysis import analysis as an
from euclid_analysis import constants as C
from euclid_analysis.statistics import basic_stats


class TestPorterError:
    def test_error_series_keys_and_type(self):
        meta = {5: {1: 1, 2: 2, 3: 1}, 6: {1: 1, 2: 1}}
        series = an.porter_error_series(meta)
        assert set(series.keys()) == {5, 6}
        assert all(isinstance(v, float) for v in series.values())

    def test_sign_summary_counts_partition(self):
        series = {1: 1.0, 2: -1.0, 3: 1.0, 4: 1.0}
        summ = an.sign_summary(series)
        assert summ["pos"][4] + summ["neg"][4] == 4
        assert summ["sum_sign"][-1] == summ["pos"][4] - summ["neg"][4]
        assert summ["prop_pos"][4] == pytest.approx(summ["pos"][4] / 4)

    def test_error_champions_are_records(self):
        series = {1: 0.0, 2: 2.0, 3: 1.0, 4: 5.0, 5: -9.0}
        champs = an.error_champions(series)
        # a=2 (first positive record), a=4 (bigger), a=5 (negative record).
        assert set(champs.keys()) == {2, 4, 5}


class TestModels:
    def test_inverse_sqrt_model(self):
        x = np.array([1.0, 4.0, 100.0])
        np.testing.assert_allclose(an.inverse_sqrt_model(x, 2.0, 3.0),
                                   [5.0, 4.0, 3.2])

    def test_fit_recovers_planted_constants(self):
        xs = np.arange(1000, 100001, 1000, dtype=float)
        truth_a, truth_b = 1.7, -0.3
        series = {int(n): truth_a / np.sqrt(n) + truth_b for n in xs}
        a, b = an.fit_subdominant_constants(series)
        assert a == pytest.approx(truth_a, abs=1e-6)
        assert b == pytest.approx(truth_b, abs=1e-6)


class TestReproducesPublishedConstants:
    def test_norton_mean_constants(self, dataset_C, dataset_B):
        terms = an.mean_variance_error_terms(basic_stats(dataset_C),
                                             basic_stats(dataset_B))
        _, B = an.fit_subdominant_constants(terms["err_mean"])
        _, B1 = an.fit_subdominant_constants(terms["err_mean1"])
        assert B == pytest.approx(C.NU_NORTON - 0.5, abs=5e-3)
        assert B1 == pytest.approx(C.NU_NORTON_COPRIME - 0.5, abs=5e-3)

    def test_variance_subdominant_difference(self, dataset_C, dataset_B):
        terms = an.mean_variance_error_terms(basic_stats(dataset_C),
                                             basic_stats(dataset_B))
        _, D = an.fit_subdominant_constants(terms["err_var"])
        _, D1 = an.fit_subdominant_constants(terms["err_var1"])
        assert (D - D1) == pytest.approx(C.DELTA_KAPPA, abs=5e-3)

    def test_second_moment_leading_coefficient(self, dataset_H1):
        c2, _, _ = an.fit_second_moment(basic_stats(dataset_H1), 1, 1000)
        assert c2 == pytest.approx(C.LAMBDA_DIXON ** 2, abs=1e-2)

    def test_error_champion_605(self, dataset_H1):
        series = an.porter_error_series(dataset_H1)
        champs = an.error_champions(series)
        assert 605 in champs
