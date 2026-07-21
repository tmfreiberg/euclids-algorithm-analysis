"""Command-line interface for :mod:`euclid_analysis`.

Nearly every public function in the package is reachable as a subcommand.  Each
command also accepts the underlying function's name as an alias, so, for
example, ``euclid`` and ``write_euclid`` both work.

Run ``euclid-analysis --help`` (or ``ea --help``) for the full list, and
``euclid-analysis <command> --help`` for a specific command.

Arguments are separated by spaces, not written as a function call::

    euclid-analysis write_euclid 7 23      # correct
    euclid-analysis write_euclid(7,23)     # WRONG (shell passes one token)

Installation wires up two console commands via ``pyproject.toml``::

    [project.scripts]
    euclid-analysis = "euclid_analysis.cli:main"
    ea              = "euclid_analysis.cli:main"

so ``ea`` is a short alias for ``euclid-analysis``.  It can also be run without
installing as ``python -m euclid_analysis.cli <command> ...``.

The one public function not exposed is ``dictionary_sort``: it merely returns a
key-sorted copy of a dictionary and has no natural command-line form.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from euclid_analysis import (
    __version__,
    analysis,
    basic_stats,
    constants,
    count_steps,
    dictionary_statistics,
    euclid_alg_frequencies,
    fibonacci,
    gcd,
    gcdn,
    heilbronn,
    quotients_remainders,
    remainders,
    save_meta_dict,
    tabulate,
    write_euclid,
)
from euclid_analysis.dataio import DATA_DIR, DATASETS, load_dataset, load_meta_dict


# --------------------------------------------------------------------------- #
# Output helpers
# --------------------------------------------------------------------------- #
def _print_freq(
    freq: Mapping[int, float], label: str | None = None, fmt: str = "d"
) -> None:
    """Print a ``{value: count}`` mapping in sorted key order."""
    if label:
        print(label)
    if not freq:
        print("  (empty)")
        return
    width = max(len(str(k)) for k in freq)
    for k in sorted(freq):
        v = freq[k]
        line = f"  {k:>{width}} : {v:d}" if fmt == "d" else f"  {k:>{width}} : {v:.6f}"
        print(line)


def _print_stats(stats: Mapping[str, Any], indent: str = "  ") -> None:
    """Print the summary produced by :func:`dictionary_statistics`."""
    print(f"{indent}mean            = {stats['mean']:.6f}")
    print(f"{indent}second moment   = {stats['2ndmom']:.6f}")
    print(f"{indent}variance        = {stats['var']:.6f}")
    print(f"{indent}std deviation   = {stats['sdv']:.6f}")
    print(f"{indent}median          = {stats['med']}")
    print(f"{indent}mode(s)         = {stats['mode']}")


def _fail(message: str) -> int:
    """Print an error to stderr and return a non-zero exit code."""
    print(f"error: {message}", file=sys.stderr)
    return 1


# --------------------------------------------------------------------------- #
# algorithms
# --------------------------------------------------------------------------- #
def cmd_gcd(a: argparse.Namespace) -> int:
    """Print gcd(a, b)."""
    print(gcd(a.a, a.b))
    return 0


def cmd_gcdn(a: argparse.Namespace) -> int:
    """Print the gcd of two or more integers."""
    if len(a.ints) < 2:
        return _fail("gcdn needs at least two integers")
    print(gcdn(*a.ints))
    return 0


def cmd_steps(a: argparse.Namespace) -> int:
    """Print gcd(a, b) and the step count T(a, b)."""
    g, s = count_steps(a.a, a.b)
    print(f"gcd({a.a}, {a.b}) = {g}")
    print(f"T({a.a}, {a.b})   = {s}")
    return 0


def cmd_remainders(a: argparse.Namespace) -> int:
    """Print the remainder sequence of Euclid's algorithm."""
    print(remainders(a.a, a.b))
    return 0


def cmd_quotients(a: argparse.Namespace) -> int:
    """Print the quotient and remainder sequences."""
    q, r = quotients_remainders(a.a, a.b)
    print(f"quotients : {q}")
    print(f"remainders: {r}")
    return 0


def cmd_euclid(a: argparse.Namespace) -> int:
    """Print a human-readable transcript of the whole computation."""
    print(write_euclid(a.a, a.b))
    return 0


# --------------------------------------------------------------------------- #
# fibonacci
# --------------------------------------------------------------------------- #
def cmd_fib(a: argparse.Namespace) -> int:
    """Print the n-th Fibonacci number (fast two-variable method)."""
    print(fibonacci.fib(a.n))
    return 0


def cmd_naive_fib(a: argparse.Namespace) -> int:
    """Print the n-th Fibonacci number by naive recursion."""
    print(fibonacci.naive_fib(a.n))
    return 0


def cmd_fibonacci_list(a: argparse.Namespace) -> int:
    """Print the list f_0 .. f_n."""
    print(fibonacci.fibonacci_list(a.n))
    return 0


def cmd_binet_fib(a: argparse.Namespace) -> int:
    """Print the n-th Fibonacci number via Binet's formula."""
    print(fibonacci.binet_fib(a.n))
    return 0


def cmd_fib_time(a: argparse.Namespace) -> int:
    """Time tabulating f_0 .. f_n and report the result."""
    values, durations = fibonacci.time_fibonacci_list(a.n)
    print(f"tabulated f_0 .. f_{a.n} in {sum(durations):.6f} s")
    print(f"f_{a.n} = {values[-1]}")
    return 0


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #
def cmd_constants(a: argparse.Namespace) -> int:
    """Print the golden ratio and the mean/variance constants."""
    c = constants
    rows = [
        ("gamma", c.EULER_MASCHERONI),
        ("zeta(2)", c.zeta(2)),
        ("zeta'(2)", c.ZETA_PRIME_2),
        ("zeta''(2)", c.ZETA_DOUBLE_PRIME_2),
        ("phi (golden ratio)", fibonacci.PHI),
        ("lambda", c.LAMBDA_DIXON),
        ("lambda^2", c.LAMBDA_DIXON**2),
        ("C_P (Porter)", c.PORTER_CONSTANT),
        ("nu (Norton)", c.NU_NORTON),
        ("nu_1 (Norton, coprime)", c.NU_NORTON_COPRIME),
        ("eta (Hensley)", c.ETA_HENSLEY),
        ("kappa - kappa_1", c.DELTA_KAPPA),
        ("kappa (guess)", c.KAPPA_VAR),
        ("kappa_1 (guess)", c.KAPPA_VAR_COPRIME),
    ]
    width = max(len(name) for name, _ in rows)
    for name, value in rows:
        print(f"{name:>{width}} = {value}")
    return 0


# --------------------------------------------------------------------------- #
# frequencies (1-D and 2-D)
# --------------------------------------------------------------------------- #
def cmd_heilbronn(a: argparse.Namespace) -> int:
    """Print 1-D step-count frequencies for a single a."""
    gcdlist = a.gcdlist if a.gcdlist else [1]
    restricted, all_ = heilbronn(gcdlist, [a.a])
    print(f"a = {a.a}")
    _print_freq(all_[a.a], "step-count frequencies over all b in [1, a]:")
    _print_freq(
        restricted[a.a],
        f"step-count frequencies over b with gcd(a, b) in {gcdlist}:",
    )
    return 0


def cmd_frequencies(a: argparse.Namespace) -> int:
    """Print 2-D step-count frequencies over pairs 0 < b < a < N."""
    gcdlist = a.gcdlist if a.gcdlist else [1]
    if a.n > a.warn_above:
        print(
            f"note: N = {a.n} sweeps ~{a.n * (a.n - 1) // 2} pairs "
            "and may take a while.",
            file=sys.stderr,
        )
    gcd_freq, steps_restricted, steps_all = euclid_alg_frequencies(
        gcdlist, [], [1, a.n], {}, {}, {}
    )
    print(f"N = {a.n} (pairs 0 < b < a < N)")
    _print_freq(steps_all[a.n], "step-count frequencies over all pairs:")
    _print_freq(
        steps_restricted[a.n],
        f"step-count frequencies over pairs with gcd in {gcdlist}:",
    )
    if a.gcd_values:
        _print_freq(gcd_freq[a.n], "gcd-value frequencies:")
    return 0


# --------------------------------------------------------------------------- #
# datasets and statistics
# --------------------------------------------------------------------------- #
def cmd_datasets(a: argparse.Namespace) -> int:
    """List the shipped datasets and whether their files are present."""
    print(f"data directory: {DATA_DIR}")
    width = max(len(name) for name in DATASETS)
    for name in sorted(DATASETS):
        path = DATA_DIR / DATASETS[name]
        present = "present" if path.exists() else "MISSING"
        print(f"  {name:>{width}} : {DATASETS[name]}  [{present}]")
    return 0


def cmd_dataset(a: argparse.Namespace) -> int:
    """Load a dataset (named or from a file) and inspect it."""
    if a.file:
        try:
            meta = load_meta_dict(a.file)
        except Exception as exc:  # noqa: BLE001 - report any load failure
            return _fail(f"could not load {a.file}: {exc}")
        source = a.file
    elif a.name:
        try:
            meta = load_dataset(a.name)
        except KeyError:
            return _fail(f"unknown dataset {a.name!r}; choose from {sorted(DATASETS)}")
        except FileNotFoundError as exc:
            return _fail(str(exc))
        source = a.name
    else:
        return _fail("give a dataset name or --file PATH")

    keys = sorted(meta)
    if a.key is None:
        print(f"dataset {source}: {len(keys)} columns, keys {keys[0]} .. {keys[-1]}")
        if a.table:
            print(tabulate(meta))
        return 0

    if a.key not in meta:
        return _fail(f"key {a.key} not in {source} (range {keys[0]} .. {keys[-1]})")
    if a.dist:
        dist = dictionary_statistics(meta[a.key])["dist"]
        _print_freq(dist, f"column {a.key} (relative frequencies):", fmt="f")
    elif a.table:
        print(tabulate({a.key: meta[a.key]}))
    else:
        _print_freq(meta[a.key], f"column {a.key}:")
    if a.stats:
        print(f"statistics for column {a.key}:")
        _print_stats(dictionary_statistics(meta[a.key]))
    return 0


def cmd_stats(a: argparse.Namespace) -> int:
    """Print per-column summary statistics for a whole dataset."""
    try:
        meta = load_dataset(a.name)
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))
    stats = basic_stats(meta)
    header = f"{'key':>8} {'mean':>10} {'variance':>10} {'std':>9} {'median':>8}  mode"
    print(header)
    for k in sorted(stats):
        s = stats[k]
        print(
            f"{k:>8} {s['mean']:>10.4f} {s['var']:>10.4f} "
            f"{s['sdv']:>9.4f} {str(s['med']):>8}  {s['mode']}"
        )
    return 0


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def cmd_porter_error(a: argparse.Namespace) -> int:
    """Print the summed Porter error E(a) for a single a."""
    try:
        meta = load_dataset(a.dataset)
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))
    if a.a not in meta:
        keys = sorted(meta)
        return _fail(f"a = {a.a} not in dataset (range {keys[0]} .. {keys[-1]})")
    print(f"E({a.a}) = {analysis.porter_error(meta[a.a], a.a):.6f}")
    return 0


def cmd_porter_series(a: argparse.Namespace) -> int:
    """Print the Porter error series E(a) over a range."""
    try:
        meta = load_dataset(a.dataset)
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))
    series = analysis.porter_error_series(meta)
    lo = a.from_ if a.from_ is not None else min(series)
    hi = a.to if a.to is not None else max(series)
    print(f"E(a) for a in [{lo}, {hi}] (dataset {a.dataset}):")
    for k in sorted(series):
        if lo <= k <= hi:
            print(f"  E({k}) = {series[k]:.6f}")
    return 0


def cmd_sign_summary(a: argparse.Namespace) -> int:
    """Print the random-walk sign summary of the Porter error series."""
    try:
        meta = load_dataset(a.dataset)
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))
    summ = analysis.sign_summary(analysis.porter_error_series(meta))
    n = max(summ["pos"])  # number of sign-bearing a (E(a) nonzero)
    print(f"Porter error sign summary (dataset {a.dataset}):")
    print(f"  sign-bearing a (E(a) nonzero) : {n}")
    print(f"  positive : {summ['pos'][n]} ({100 * summ['prop_pos'][n]:.2f}%)")
    print(f"  negative : {summ['neg'][n]} ({100 * summ['prop_neg'][n]:.2f}%)")
    print(f"  final signed sum (pos - neg walk) : {summ['sum_sign'][-1]}")
    return 0


def cmd_champions(a: argparse.Namespace) -> int:
    """Print the Porter error champions."""
    try:
        meta = load_dataset(a.dataset)
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))
    champs = analysis.error_champions(analysis.porter_error_series(meta))
    print(f"error champions from dataset {a.dataset} ({len(champs)} total):")
    for k in sorted(champs):
        err, _c_pos, _c_neg, c = champs[k]
        print(f"  a = {k:>6} : E(a) = {err:12.4f}, c = {c:.4f}")
    return 0


def cmd_fit_second_moment(a: argparse.Namespace) -> int:
    """Fit and print the second-moment coefficients c2, c1, c0."""
    try:
        meta = load_dataset(a.dataset)
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))
    c2, c1, c0 = analysis.fit_second_moment(basic_stats(meta), a.a_min, a.a_max)
    print(f"second-moment fit c2*(log a)^2 + c1*log a + c0 (dataset {a.dataset}):")
    print(f"  c2 = {c2:.4f}   (lambda^2 = {constants.LAMBDA_DIXON**2:.4f})")
    print(f"  c1 = {c1:.4f}")
    print(f"  c0 = {c0:.4f}")
    return 0


def cmd_variance_error(a: argparse.Namespace) -> int:
    """Print residuals Var(Z) - (eta*log a + constant)."""
    try:
        meta = load_dataset(a.dataset)
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))
    residuals = analysis.variance_guess_error(basic_stats(meta), a.constant)
    if a.at is not None:
        if a.at not in residuals:
            return _fail(f"a = {a.at} not in dataset")
        print(
            f"Var(Z) - (eta*log a + {a.constant}) at a = {a.at}: {residuals[a.at]:.6f}"
        )
        return 0
    vals = list(residuals.values())
    print(f"variance-estimate residuals (dataset {a.dataset}, constant {a.constant}):")
    print(f"  a in {min(residuals)} .. {max(residuals)}")
    print(
        f"  min = {min(vals):.4f}, max = {max(vals):.4f}, "
        f"mean = {sum(vals) / len(vals):.4f}"
    )
    return 0


def cmd_error_terms(a: argparse.Namespace) -> int:
    """Print a chosen 2-D error series (mean/variance, all/coprime)."""
    try:
        cstats = basic_stats(load_dataset("C"))
        bstats = basic_stats(load_dataset("B"))
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))
    series = analysis.mean_variance_error_terms(cstats, bstats)
    chosen = series[a.which]
    lo = a.from_ if a.from_ is not None else min(chosen)
    hi = a.to if a.to is not None else max(chosen)
    print(f"{a.which} for N in [{lo}, {hi}]:")
    for n in sorted(chosen):
        if lo <= n <= hi:
            print(f"  N = {n:>6} : {chosen[n]:.6f}")
    return 0


def cmd_fit_constants(a: argparse.Namespace) -> int:
    """Fit and print the subdominant constants B, B_1, D, D_1."""
    try:
        cstats = basic_stats(load_dataset("C"))
        bstats = basic_stats(load_dataset("B"))
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))
    series = analysis.mean_variance_error_terms(cstats, bstats)
    _, b = analysis.fit_subdominant_constants(series["err_mean"], a.n_min, a.n_max)
    _, b1 = analysis.fit_subdominant_constants(series["err_mean1"], a.n_min, a.n_max)
    _, d = analysis.fit_subdominant_constants(series["err_var"], a.n_min, a.n_max)
    _, d1 = analysis.fit_subdominant_constants(series["err_var1"], a.n_min, a.n_max)
    print("subdominant constants fitted from the shipped data:")
    print(f"  B   (mean, all pairs)     = {b:+.4f}")
    print(f"  B_1 (mean, coprime)       = {b1:+.4f}")
    print(f"  D   (variance, all pairs) = {d:+.4f}")
    print(f"  D_1 (variance, coprime)   = {d1:+.4f}")
    print(f"  B_1 - B  = {b1 - b:+.4f}   (cf. nu_1 - nu = 0.4803)")
    print(f"  D  - D_1 = {d - d1:+.4f}   (cf. kappa - kappa_1 = 0.3340)")
    return 0


def cmd_model_second_moment(a: argparse.Namespace) -> int:
    """Evaluate c2*(log x)^2 + c1*log x + c0 at a point."""
    import numpy as np

    val = analysis.second_moment_model(np.array([a.x], dtype=float), a.c2, a.c1, a.c0)
    print(f"{float(val[0]):.6f}")
    return 0


def cmd_model_inverse_sqrt(a: argparse.Namespace) -> int:
    """Evaluate a/sqrt(x) + b at a point."""
    import numpy as np

    val = analysis.inverse_sqrt_model(np.array([a.x], dtype=float), a.a_coef, a.b_coef)
    print(f"{float(val[0]):.6f}")
    return 0


# --------------------------------------------------------------------------- #
# data generation (slow)
# --------------------------------------------------------------------------- #
def cmd_generate_1d(a: argparse.Namespace) -> int:
    """Generate and save the 1-D frequency tables."""
    from pathlib import Path

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    restricted, all_ = heilbronn([1], list(range(1, a.a_max + 1)))
    save_meta_dict(restricted, out / f"euclid_steps_1d-coprime-{a.a_max + 1}df.csv")
    save_meta_dict(all_, out / f"euclid_steps_1d-all-{a.a_max + 1}df.csv")
    print(f"wrote 1-D coprime and all tables (a up to {a.a_max}) to {out}")
    return 0


def cmd_generate_2d(a: argparse.Namespace) -> int:
    """Generate and save the 2-D frequency tables."""
    from pathlib import Path

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    checkpoints = list(range(1, a.n + 1, a.step))
    if checkpoints[-1] != a.n:
        checkpoints.append(a.n)
    gcd_freq, steps_coprime, steps_all = euclid_alg_frequencies(
        [1], [], checkpoints, {}, {}, {}
    )
    save_meta_dict(gcd_freq, out / f"gcd_pairs_{a.n}df.csv")
    save_meta_dict(steps_coprime, out / f"euclid_steps_coprime_{a.n}df.csv")
    save_meta_dict(steps_all, out / f"euclid_steps_{a.n}df.csv")
    print(f"wrote 2-D gcd, coprime, and all tables (N up to {a.n}) to {out}")
    return 0


# --------------------------------------------------------------------------- #
# plotting (needs matplotlib)
# --------------------------------------------------------------------------- #
def cmd_plot(a: argparse.Namespace) -> int:
    """Render a figure or animation from the shipped data."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return _fail("plotting needs matplotlib ('pip install matplotlib')")

    from euclid_analysis import dists, plotting

    try:
        if a.kind in {"1d-dist", "2d-dist"}:
            if a.kind == "1d-dist":
                h1 = load_dataset("H1")
                fig, anim = plotting.animate_one_dim_distribution(
                    dists(h1), basic_stats(h1)
                )
            else:
                b, call = load_dataset("B"), load_dataset("C")
                fig, anim = plotting.animate_two_dim_distribution(
                    dists(call), basic_stats(call), dists(b), basic_stats(b)
                )
            plotting.save_gif(anim, a.out)
            plt.close(fig)
            print(f"wrote animation to {a.out}")
            return 0

        fig, ax = plt.subplots()
        if a.kind == "porter-error":
            h1 = load_dataset("H1")
            series = analysis.porter_error_series(h1)
            plotting.plot_porter_error_frame(
                ax, series, analysis.error_champions(series), a.at
            )
        elif a.kind == "porter-sign":
            h1 = load_dataset("H1")
            summary = analysis.sign_summary(analysis.porter_error_series(h1))
            plotting.plot_porter_sign_sum_frame(ax, summary, a.at)
        elif a.kind == "second-moment":
            plotting.plot_second_moment_frame(ax, basic_stats(load_dataset("H1")), a.at)
        elif a.kind == "variance-error":
            h1 = load_dataset("H1")
            plotting.plot_variance_error_frame(
                ax, analysis.variance_guess_error(basic_stats(h1)), a.at
            )
        fig.savefig(a.out, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"wrote figure to {a.out}")
        return 0
    except (KeyError, FileNotFoundError) as exc:
        return _fail(str(exc))


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with every subcommand and its aliases."""
    parser = argparse.ArgumentParser(
        prog="euclid-analysis",
        description="Terminal access to the euclid_analysis package (alias: ea).",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    def add(
        name: str,
        func: Any,
        help_text: str,
        aliases: list[str] | None = None,
    ) -> argparse.ArgumentParser:
        """Register a subcommand and return its parser."""
        p = sub.add_parser(
            name, aliases=aliases or [], help=help_text, description=help_text
        )
        p.set_defaults(func=func)
        return p

    def ab(p: argparse.ArgumentParser) -> None:
        """Add the two positional integers a and b."""
        p.add_argument("a", type=int, help="first integer")
        p.add_argument("b", type=int, help="second integer")

    def one_n(p: argparse.ArgumentParser, helptxt: str = "index") -> None:
        """Add the single positional integer n."""
        p.add_argument("n", type=int, help=helptxt)

    # -- algorithms --
    ab(add("gcd", cmd_gcd, "greatest common divisor of a and b"))
    p = add("gcdn", cmd_gcdn, "greatest common divisor of two or more integers")
    p.add_argument("ints", nargs="+", type=int, metavar="int", help="two or more")
    ab(
        add(
            "steps",
            cmd_steps,
            "gcd(a, b) and the step count T(a, b)",
            aliases=["count_steps", "gcd_steps"],
        )
    )
    ab(add("remainders", cmd_remainders, "the remainder sequence"))
    ab(
        add(
            "quotients",
            cmd_quotients,
            "the quotient and remainder sequences",
            aliases=["quotients_remainders"],
        )
    )
    ab(
        add(
            "euclid",
            cmd_euclid,
            "a human-readable transcript of the whole computation",
            aliases=["write_euclid"],
        )
    )

    # -- fibonacci --
    one_n(add("fib", cmd_fib, "the n-th Fibonacci number (fast two-variable method)"))
    one_n(
        add(
            "naive-fib",
            cmd_naive_fib,
            "the n-th Fibonacci number by naive recursion (exponential!)",
            aliases=["naive_fib"],
        )
    )
    one_n(
        add(
            "fibonacci-list",
            cmd_fibonacci_list,
            "the list f_0 .. f_n",
            aliases=["fibonacci_list", "fib-list"],
        )
    )
    one_n(
        add(
            "binet-fib",
            cmd_binet_fib,
            "the n-th Fibonacci number via Binet's formula",
            aliases=["binet_fib"],
        )
    )
    one_n(
        add(
            "fib-time",
            cmd_fib_time,
            "time tabulating f_0 .. f_n",
            aliases=["time_fibonacci_list"],
        )
    )

    # -- constants --
    add("constants", cmd_constants, "print the golden ratio and the constants")

    # -- frequencies --
    p = add("heilbronn", cmd_heilbronn, "1-D step-count frequencies for a single a")
    p.add_argument("a", type=int, help="the value of a")
    p.add_argument(
        "--gcdlist", nargs="+", type=int, metavar="g", help="admitted gcds (def: 1)"
    )

    p = add(
        "frequencies",
        cmd_frequencies,
        "2-D step-count frequencies over pairs 0 < b < a < N",
        aliases=["euclid_alg_frequencies"],
    )
    p.add_argument("n", type=int, help="upper bound N")
    p.add_argument(
        "--gcdlist", nargs="+", type=int, metavar="g", help="admitted gcds (def: 1)"
    )
    p.add_argument(
        "--gcd-values", action="store_true", help="also print gcd-value frequencies"
    )
    p.add_argument(
        "--warn-above", type=int, default=2000, help="warn when N exceeds this"
    )

    # -- datasets / statistics --
    add("datasets", cmd_datasets, "list the shipped datasets and their presence")

    p = add(
        "dataset",
        cmd_dataset,
        "load a dataset (by name or --file) and inspect it",
        aliases=["load_dataset", "load_meta_dict"],
    )
    p.add_argument("name", nargs="?", help=f"dataset name, one of {sorted(DATASETS)}")
    p.add_argument("--file", help="load an arbitrary frequency-table CSV instead")
    p.add_argument("--key", type=int, help="show a single column (a or N)")
    p.add_argument("--stats", action="store_true", help="also print summary statistics")
    p.add_argument(
        "--dist", action="store_true", help="show relative frequencies not counts"
    )
    p.add_argument("--table", action="store_true", help="print as a pandas DataFrame")

    p = add(
        "stats",
        cmd_stats,
        "per-column summary statistics for a whole dataset",
        aliases=["basic_stats"],
    )
    p.add_argument("name", help=f"dataset name, one of {sorted(DATASETS)}")

    # -- analysis --
    p = add(
        "porter-error",
        cmd_porter_error,
        "the summed Porter error E(a) for a single a",
        aliases=["porter_error"],
    )
    p.add_argument("a", type=int, help="the value of a")
    p.add_argument("--dataset", default="H1", help="coprime 1-D dataset (def: H1)")

    p = add(
        "porter-series",
        cmd_porter_series,
        "the Porter error series E(a) over a range",
        aliases=["porter_error_series"],
    )
    p.add_argument("--dataset", default="H1", help="coprime 1-D dataset (def: H1)")
    p.add_argument("--from", dest="from_", type=int, default=None, help="lower a")
    p.add_argument("--to", type=int, default=None, help="upper a")

    p = add(
        "sign-summary",
        cmd_sign_summary,
        "random-walk sign summary of the Porter error series",
        aliases=["sign_summary"],
    )
    p.add_argument("--dataset", default="H1", help="coprime 1-D dataset (def: H1)")

    p = add(
        "champions",
        cmd_champions,
        "Porter error champions (record values of E(a)/sqrt(a))",
        aliases=["error_champions"],
    )
    p.add_argument("--dataset", default="H1", help="coprime 1-D dataset (def: H1)")

    p = add(
        "fit-second-moment",
        cmd_fit_second_moment,
        "fit E[Z^2] to c2*(log a)^2 + c1*log a + c0",
        aliases=["fit_second_moment"],
    )
    p.add_argument("--dataset", default="H1", help="1-D dataset (def: H1)")
    p.add_argument("--a-min", type=int, default=1, help="lower a for the fit")
    p.add_argument("--a-max", type=int, default=None, help="upper a for the fit")

    p = add(
        "variance-error",
        cmd_variance_error,
        "residuals Var(Z) - (eta*log a + constant)",
        aliases=["variance_guess_error"],
    )
    p.add_argument("--dataset", default="H1", help="1-D dataset (def: H1)")
    p.add_argument("--at", type=int, default=None, help="residual at this a")
    p.add_argument(
        "--constant", type=float, default=-0.354293955, help="subtracted constant"
    )

    p = add(
        "error-terms",
        cmd_error_terms,
        "a 2-D error series (mean/variance, all/coprime)",
        aliases=["mean_variance_error_terms"],
    )
    p.add_argument(
        "--which",
        choices=[
            "err_mean",
            "err_mean1",
            "err_var",
            "err_var1",
            "mean_delta",
            "var_delta",
        ],
        default="err_var",
        help="which series to print (def: err_var)",
    )
    p.add_argument("--from", dest="from_", type=int, default=None, help="lower N")
    p.add_argument("--to", type=int, default=None, help="upper N")

    p = add(
        "fit-constants",
        cmd_fit_constants,
        "fit B, B_1, D, D_1 from the shipped 2-D data",
        aliases=["fit_subdominant_constants"],
    )
    p.add_argument("--n-min", type=int, default=None, help="lower checkpoint")
    p.add_argument("--n-max", type=int, default=None, help="upper checkpoint")

    p = add(
        "model-second-moment",
        cmd_model_second_moment,
        "evaluate c2*(log x)^2 + c1*log x + c0",
        aliases=["second_moment_model"],
    )
    p.add_argument("x", type=float, help="the point x")
    p.add_argument("c2", type=float)
    p.add_argument("c1", type=float)
    p.add_argument("c0", type=float)

    p = add(
        "model-inverse-sqrt",
        cmd_model_inverse_sqrt,
        "evaluate a/sqrt(x) + b",
        aliases=["inverse_sqrt_model"],
    )
    p.add_argument("x", type=float, help="the point x")
    p.add_argument("a_coef", type=float, metavar="a")
    p.add_argument("b_coef", type=float, metavar="b")

    # -- generation --
    p = add("generate-1d", cmd_generate_1d, "generate and save the 1-D tables (slow)")
    p.add_argument("a_max", type=int, help="largest a to tabulate")
    p.add_argument("--out", default="data", help="output directory (def: data)")

    p = add(
        "generate-2d", cmd_generate_2d, "generate and save the 2-D tables (very slow)"
    )
    p.add_argument("n", type=int, help="largest checkpoint N")
    p.add_argument("--step", type=int, default=1000, help="checkpoint spacing")
    p.add_argument("--out", default="data", help="output directory (def: data)")

    # -- plotting --
    p = add("plot", cmd_plot, "render a figure or animation (needs matplotlib)")
    p.add_argument(
        "kind",
        choices=[
            "porter-error",
            "porter-sign",
            "second-moment",
            "variance-error",
            "1d-dist",
            "2d-dist",
        ],
        help="which figure to draw",
    )
    p.add_argument("--out", required=True, help="output file (.png frame, .gif anim)")
    p.add_argument("--at", type=int, default=1000, help="frame index N (def: 1000)")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the chosen command."""
    try:
        args = build_parser().parse_args(argv)
        return args.func(args)
    except BrokenPipeError:
        # A downstream reader (e.g. `head`) closed the pipe; exit quietly.
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0


if __name__ == "__main__":
    sys.exit(main())