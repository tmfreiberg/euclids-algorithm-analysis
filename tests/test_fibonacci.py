"""Tests for :mod:`euclid_analysis.fibonacci`."""

from __future__ import annotations

import pytest

from euclid_analysis import fibonacci as fibmod

FIRST_FIFTEEN = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377]


class TestFib:
    def test_prefix_matches_reference(self):
        assert [fibmod.fib(n) for n in range(15)] == FIRST_FIFTEEN

    def test_negative_index_is_reflected(self):
        for n in range(1, 15):
            assert fibmod.fib(-n) == fibmod.fib(n)

    def test_recurrence(self):
        for n in range(2, 50):
            assert fibmod.fib(n) == fibmod.fib(n - 1) + fibmod.fib(n - 2)


class TestNaiveFib:
    def test_matches_fib_on_small_inputs(self):
        for n in range(0, 20):
            assert fibmod.naive_fib(n) == fibmod.fib(n)

    def test_negative_index_is_reflected(self):
        assert fibmod.naive_fib(-10) == fibmod.naive_fib(10)


class TestFibonacciList:
    def test_returns_prefix(self):
        assert fibmod.fibonacci_list(14) == FIRST_FIFTEEN

    def test_length_is_n_plus_one(self):
        assert len(fibmod.fibonacci_list(0)) == 1
        assert len(fibmod.fibonacci_list(30)) == 31

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            fibmod.fibonacci_list(-1)


class TestBinetFib:
    def test_matches_fib_within_float_range(self):
        for n in range(0, 70):
            assert fibmod.binet_fib(n) == fibmod.fib(n)


class TestTiming:
    def test_returns_values_and_durations(self):
        values, durations = fibmod.time_fibonacci_list(20)
        assert values == fibmod.fibonacci_list(20)
        assert len(durations) == 21
        assert all(d >= 0 for d in durations)
