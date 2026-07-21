"""Tests for the :mod:`euclid_analysis.cli` command-line interface.

Each test drives the CLI through ``cli.main(argv)`` and captures stdout/stderr
with the ``capsys`` fixture.  Commands that need the shipped CSV tables request
the dataset fixtures from ``conftest.py``, which skip when a file is absent.
"""

from __future__ import annotations

import pytest

from euclid_analysis import cli


def run(capsys, *argv):
    """Invoke the CLI and return ``(exit_code, stdout, stderr)``."""
    code = cli.main(list(argv))
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# --------------------------------------------------------------------------- #
# algorithms
# --------------------------------------------------------------------------- #
def test_gcd(capsys):
    code, out, _ = run(capsys, "gcd", "1011", "69")
    assert code == 0
    assert out.strip() == "3"


def test_gcdn(capsys):
    _, out, _ = run(capsys, "gcdn", "12", "18", "30")
    assert out.strip() == "6"


def test_gcdn_too_few_args(capsys):
    code, _, err = run(capsys, "gcdn", "5")
    assert code == 1
    assert "at least two" in err


def test_steps(capsys):
    _, out, _ = run(capsys, "steps", "1011", "69")
    assert "gcd(1011, 69) = 3" in out
    assert "T(1011, 69)   = 5" in out


def test_remainders(capsys):
    _, out, _ = run(capsys, "remainders", "1011", "69")
    assert out.strip() == "[1011, 69, 45, 24, 21, 3]"


def test_quotients(capsys):
    _, out, _ = run(capsys, "quotients", "1011", "69")
    assert "quotients : [14, 1, 1, 1, 7]" in out
    assert "remainders: [1011, 69, 45, 24, 21, 3]" in out


def test_euclid_transcript(capsys):
    _, out, _ = run(capsys, "euclid", "1011", "69")
    assert "gcd(1011,69) = 3, T(1011,69) = 5" in out


# --------------------------------------------------------------------------- #
# aliases resolve to the same handlers
# --------------------------------------------------------------------------- #
def test_write_euclid_alias(capsys):
    _, out, _ = run(capsys, "write_euclid", "1011", "69")
    assert "gcd(1011,69) = 3, T(1011,69) = 5" in out


def test_count_steps_alias(capsys):
    _, out, _ = run(capsys, "count_steps", "1011", "69")
    assert "gcd(1011, 69) = 3" in out


# --------------------------------------------------------------------------- #
# fibonacci
# --------------------------------------------------------------------------- #
def test_fib(capsys):
    _, out, _ = run(capsys, "fib", "10")
    assert out.strip() == "55"


@pytest.mark.parametrize(
    "command", ["naive-fib", "binet-fib", "naive_fib", "binet_fib"]
)
def test_fib_variants(capsys, command):
    _, out, _ = run(capsys, command, "10")
    assert out.strip() == "55"


def test_fibonacci_list(capsys):
    _, out, _ = run(capsys, "fibonacci-list", "6")
    assert out.strip() == "[0, 1, 1, 2, 3, 5, 8]"


def test_fib_time(capsys):
    code, out, _ = run(capsys, "fib-time", "50")
    assert code == 0
    assert "f_50 =" in out


# --------------------------------------------------------------------------- #
# constants and frequencies
# --------------------------------------------------------------------------- #
def test_constants(capsys):
    _, out, _ = run(capsys, "constants")
    assert "phi (golden ratio)" in out
    assert "lambda" in out
    assert "eta (Hensley)" in out


def test_heilbronn(capsys):
    code, out, _ = run(capsys, "heilbronn", "5")
    assert code == 0
    assert "a = 5" in out


def test_frequencies(capsys):
    _, out, _ = run(capsys, "frequencies", "20")
    assert "N = 20" in out


# --------------------------------------------------------------------------- #
# datasets and model evaluators
# --------------------------------------------------------------------------- #
def test_datasets(capsys):
    _, out, _ = run(capsys, "datasets")
    assert "data directory" in out


def test_dataset_unknown(capsys):
    code, _, err = run(capsys, "dataset", "ZZZ")
    assert code == 1
    assert "unknown dataset" in err


def test_model_inverse_sqrt(capsys):
    _, out, _ = run(capsys, "model-inverse-sqrt", "10000", "1.5", "-0.43")
    assert out.strip() == "-0.415000"


def test_model_second_moment(capsys):
    code, out, _ = run(capsys, "model-second-moment", "100", "0.71", "1.3", "-0.5")
    assert code == 0
    float(out.strip())  # output parses as a number


# --------------------------------------------------------------------------- #
# top-level flags
# --------------------------------------------------------------------------- #
def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--help"])
    assert exc.value.code == 0


def test_version_exits_zero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_no_command_errors():
    with pytest.raises(SystemExit) as exc:
        cli.main([])
    assert exc.value.code != 0


# --------------------------------------------------------------------------- #
# dataset-dependent commands (skip if the shipped CSVs are absent)
# --------------------------------------------------------------------------- #
def test_stats(capsys, dataset_C):
    code, out, _ = run(capsys, "stats", "C")
    assert code == 0
    assert "mean" in out


def test_dataset_key_stats(capsys, dataset_C):
    code, out, _ = run(capsys, "dataset", "C", "--key", "1001", "--stats")
    assert code == 0
    assert "mean" in out


def test_porter_error(capsys, dataset_H1):
    code, out, _ = run(capsys, "porter-error", "605")
    assert code == 0
    assert "E(605)" in out


def test_champions(capsys, dataset_H1):
    code, out, _ = run(capsys, "champions")
    assert code == 0
    assert "error champions" in out


def test_sign_summary(capsys, dataset_H1):
    code, out, _ = run(capsys, "sign-summary")
    assert code == 0
    assert "positive" in out


def test_fit_constants(capsys, dataset_B, dataset_C):
    code, out, _ = run(capsys, "fit-constants")
    assert code == 0
    assert "-0.4357" in out


def test_plot_frame(capsys, dataset_H1, tmp_path):
    out_file = tmp_path / "porter.png"
    code, _, _ = run(
        capsys, "plot", "porter-error", "--at", "650", "--out", str(out_file)
    )
    assert code == 0
    assert out_file.exists()


# --------------------------------------------------------------------------- #
# generation to a temporary directory (fast, small)
# --------------------------------------------------------------------------- #
def test_generate_1d(capsys, tmp_path):
    code, _, _ = run(capsys, "generate-1d", "10", "--out", str(tmp_path))
    assert code == 0
    assert (tmp_path / "euclid_steps_1d-all-11df.csv").exists()
    assert (tmp_path / "euclid_steps_1d-coprime-11df.csv").exists()