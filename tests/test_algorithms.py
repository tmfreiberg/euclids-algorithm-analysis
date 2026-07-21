"""Tests for :mod:`euclid_analysis.algorithms`."""

from __future__ import annotations

import math

import pytest

from euclid_analysis import algorithms as alg
from euclid_analysis.fibonacci import fib


class TestGcd:
    @pytest.mark.parametrize(
        ("a", "b", "expected"),
        [
            (1011, 69, 3),
            (69, 1011, 3),
            (67737, 4623, 201),
            (17, 5, 1),
            (100, 10, 10),
            (0, 0, 0),
            (7, 0, 7),
            (0, 7, 7),
        ],
    )
    def test_known_values(self, a, b, expected):
        assert alg.gcd(a, b) == expected

    @pytest.mark.parametrize(
        ("a", "b"), [(1011, 69), (-1011, -69), (1011, -69), (-1011, 69)]
    )
    def test_sign_invariance(self, a, b):
        assert alg.gcd(a, b) == alg.gcd(abs(a), abs(b))

    def test_agrees_with_math_gcd(self):
        for a in range(0, 60):
            for b in range(0, 60):
                assert alg.gcd(a, b) == math.gcd(a, b)


class TestGcdn:
    def test_examples(self):
        assert alg.gcdn(12, 18, 30) == 6
        assert alg.gcdn(7, 14) == 7
        assert alg.gcdn(2, 3, 4, 5) == 1

    def test_matches_pairwise_reduction(self):
        assert alg.gcdn(24, 36, 48, 60) == math.gcd(math.gcd(math.gcd(24, 36), 48), 60)

    def test_requires_two_arguments(self):
        with pytest.raises(TypeError):
            alg.gcdn(5)


class TestStepCount:
    @pytest.mark.parametrize(
        ("a", "b", "steps"),
        [(1011, 69, 5), (55, 34, 8), (3, 2, 2), (5, 3, 3), (5, 0, 0), (10, 5, 1)],
    )
    def test_known_step_counts(self, a, b, steps):
        g, s = alg.count_steps(a, b)
        assert s == steps
        assert g == alg.gcd(a, b)

    def test_gcd_steps_default_arg_matches_count_steps(self):
        for a in range(1, 40):
            for b in range(0, 40):
                assert alg.gcd_steps(a, b) == alg.count_steps(a, b)

    def test_scaling_invariance_of_step_count(self):
        # Proposition 4.1(c): T(da, db) == T(a, b).
        for d in (2, 3, 67):
            assert alg.count_steps(d * 1011, d * 69)[1] == alg.count_steps(1011, 69)[1]

    def test_swap_relation(self):
        # Proposition 4.1(b): for a > b >= 1, T(b, a) == T(a, b) + 1.
        for a, b in [(1011, 69), (55, 34), (100, 7)]:
            assert alg.count_steps(b, a)[1] == alg.count_steps(a, b)[1] + 1

    def test_one_step_iff_divides(self):
        # Proposition 4.1(a): T(a, b) == 1 iff b | a (for a >= b >= 1).
        for a in range(2, 50):
            for b in range(1, a + 1):
                one_step = alg.count_steps(a, b)[1] == 1
                assert one_step == (a % b == 0)

    def test_fibonacci_worst_case(self):
        # Proposition 4.3(a): T(f_{n+2}, f_{n+1}) == n.
        for n in range(1, 20):
            assert alg.count_steps(fib(n + 2), fib(n + 1))[1] == n


class TestRemaindersAndQuotients:
    def test_remainders_reference(self):
        assert alg.remainders(1011, 69) == [1011, 69, 45, 24, 21, 3]

    def test_step_count_equals_remainder_length_minus_one(self):
        for a in range(1, 60):
            for b in range(0, 60):
                assert len(alg.remainders(a, b)) - 1 == alg.count_steps(a, b)[1]

    def test_gcd_is_last_remainder(self):
        for a in range(1, 60):
            for b in range(1, 60):
                assert abs(alg.remainders(a, b)[-1]) == alg.gcd(a, b)

    def test_quotients_remainders_reference(self):
        q, r = alg.quotients_remainders(1011, 69)
        assert q == [14, 1, 1, 1, 7]
        assert r == [1011, 69, 45, 24, 21, 3]

    def test_division_identity_holds(self):
        q, r = alg.quotients_remainders(1011, 69)
        # r[i-1] == q[i-1]*r[i] + r[i+1] with the terminal 0.
        tail = r[2:] + [0]
        for i in range(len(q)):
            assert r[i] == q[i] * r[i + 1] + tail[i]

    def test_default_args_do_not_leak_between_calls(self):
        first = alg.remainders(1011, 69)
        second = alg.remainders(55, 34)
        assert first == [1011, 69, 45, 24, 21, 3]
        assert second == [55, 34, 21, 13, 8, 5, 3, 2, 1]


class TestWriteEuclid:
    def test_returns_string_with_summary(self):
        text = alg.write_euclid(1011, 69)
        assert isinstance(text, str)
        assert "gcd(1011,69) = 3, T(1011,69) = 5" in text

    def test_first_line_reference(self):
        text = alg.write_euclid(1011, 69)
        expected = "1011 = 14*69 + 45 \t ∴ gcd(1011,69) = gcd(69,45)"
        assert text.splitlines()[0] == expected

    def test_negative_operands_rendered_with_parentheses_and_minus(self):
        text = alg.write_euclid(-1011, -69)
        assert "(-69)" in text
        assert "- 45" in text
        assert "gcd(-1011,-69) = 3, T(-1011,-69) = 5" in text
