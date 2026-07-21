"""Tests for :mod:`euclid_analysis.frequencies`."""

from __future__ import annotations

from euclid_analysis import frequencies as fr
from euclid_analysis.algorithms import count_steps


class TestDictionarySort:
    def test_orders_keys(self):
        assert list(fr.dictionary_sort({3: "c", 1: "a", 2: "b"}).keys()) == [1, 2, 3]

    def test_does_not_mutate_input(self):
        original = {3: "c", 1: "a"}
        fr.dictionary_sort(original)
        assert list(original.keys()) == [3, 1]


class TestHeilbronn:
    def test_small_case_matches_brute_force(self):
        restricted, all_ = fr.heilbronn([1], list(range(1, 21)))
        for a in range(1, 21):
            brute_all: dict[int, int] = {}
            brute_cop: dict[int, int] = {}
            for b in range(1, a + 1):
                g, s = count_steps(a, b)
                brute_all[s] = brute_all.get(s, 0) + 1
                if g == 1:
                    brute_cop[s] = brute_cop.get(s, 0) + 1
            assert all_[a] == brute_all
            assert restricted[a] == brute_cop

    def test_totals_equal_a(self):
        _, all_ = fr.heilbronn([1], [30])
        assert sum(all_[30].values()) == 30

    def test_does_not_mutate_input_list(self):
        a_list = [5, 3, 1]
        fr.heilbronn([1], a_list)
        assert a_list == [5, 3, 1]


class TestEuclidAlgFrequencies:
    def test_columns_match_brute_force(self):
        A, B, C = fr.euclid_alg_frequencies([1], [], [1, 50], {}, {}, {})

        brute_gcd: dict[int, int] = {}
        brute_all: dict[int, int] = {}
        brute_cop: dict[int, int] = {}
        for b in range(1, 50):
            for a in range(b + 1, 50):
                g, s = count_steps(a, b)
                brute_gcd[g] = brute_gcd.get(g, 0) + 1
                brute_all[s] = brute_all.get(s, 0) + 1
                if g == 1:
                    brute_cop[s] = brute_cop.get(s, 0) + 1

        assert A[50] == fr.dictionary_sort(brute_gcd)
        assert C[50] == fr.dictionary_sort(brute_all)
        assert B[50] == fr.dictionary_sort(brute_cop)

    def test_incremental_extension_matches_fresh(self):
        # Fresh computation straight to N = 60.
        A_fresh, B_fresh, C_fresh = fr.euclid_alg_frequencies(
            [1], [], [1, 60], {}, {}, {}
        )
        # Two-stage: 1 -> 30, then extend 30 -> 60.
        A1, B1, C1 = fr.euclid_alg_frequencies([1], [], [1, 30], {}, {}, {})
        A2, B2, C2 = fr.euclid_alg_frequencies([1], [1, 30], [30, 60], A1, B1, C1)
        assert A2[60] == A_fresh[60]
        assert B2[60] == B_fresh[60]
        assert C2[60] == C_fresh[60]

    def test_does_not_mutate_input_dicts(self):
        seed_A: dict = {}
        list2 = [1, 40]
        fr.euclid_alg_frequencies([1], [], list2, seed_A, {}, {})
        assert seed_A == {}
        assert list2 == [1, 40]

    def test_empty_gcdlist_admits_all(self):
        _, B, C = fr.euclid_alg_frequencies([], [], [1, 40], {}, {}, {})
        # With no gcd restriction, the restricted table equals the all table.
        assert B[40] == C[40]
