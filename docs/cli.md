# Command-line interface

`euclid_analysis` installs a command-line tool that exposes nearly every
function in the package, so you can run the algorithm, inspect the shipped
datasets, and reproduce the numerical fits without writing a script.

## Setup

Installing the package creates two equivalent commands, `euclid-analysis` and
the short alias `ea`:

```bash
python -m pip install -e .
ea --help              # the full list of commands
ea <command> --help    # options for one command
```

Without installing, the same tool runs as `python -m euclid_analysis.cli ...`.

Arguments are plain words, **not** a function call:

```bash
ea write_euclid 7 23      # correct
ea write_euclid(7,23)     # wrong: the shell passes this as a single token
```

Every command also answers to the underlying function's name, so `ea euclid`
and `ea write_euclid` do the same thing, as do `ea steps` and `ea count_steps`.

## What it covers

Run `ea --help` for the authoritative list; the commands fall into these
groups.

**The algorithm.** `gcd`, `gcdn`, `steps` (the gcd and the step count
`T(a, b)`), `remainders`, `quotients`, and `euclid` (a full aligned transcript).

```bash
$ ea euclid 1011 69
1011 = 14*69 + 45 	 ∴ gcd(1011,69) = gcd(69,45)
  69 =  1*45 + 24 	 ∴   gcd(69,45) = gcd(45,24)
  ...
gcd(1011,69) = 3, T(1011,69) = 5
```

**Fibonacci (the worst case).** `fib`, `naive-fib` (the exponential version),
`fibonacci-list`, `binet-fib`, and `fib-time`.

**Constants.** `ea constants` prints the golden ratio and the mean/variance
constants (`λ`, `C_P`, `ν`, `ν₁`, `η`, `κ − κ₁`, and the guesses for `κ`, `κ₁`).

**Frequency tables.** `heilbronn A` (one-dimensional, a single `a`) and
`frequencies N` (two-dimensional, over pairs `0 < b < a < N`). Both restrict to
coprime pairs by default; pass `--gcdlist g ...` to change the admitted gcd
values.

**Datasets.** `datasets` lists the shipped tables and whether each file is
present; `dataset NAME` inspects one (with `--key K`, `--stats`, `--dist`,
`--table`, or `--file PATH` for an arbitrary CSV); and `stats NAME` summarises
every column.

```bash
$ ea dataset C --key 1001 --stats
column 1001:
  ...
statistics for column 1001:
  mean            = 5.397922
  variance        = 3.414631
```

**Numerical analysis.** `porter-error`, `porter-series`, `sign-summary`,
`champions`, `fit-second-moment`, `variance-error`, `error-terms`, and
`fit-constants` (the subdominant constants `B, B₁, D, D₁`). The
one-dimensional commands read the `H1` dataset by default, selectable with
`--dataset`.

```bash
$ ea fit-constants
subdominant constants fitted from the shipped data:
  B   (mean, all pairs)     = -0.4357
  B_1 (mean, coprime)       = +0.0457
  D   (variance, all pairs) = -0.0950
  D_1 (variance, coprime)   = -0.4333
  ...
```

**Generating data and figures.** `generate-1d A_MAX` and `generate-2d N`
rebuild and save the CSV tables (slow); `plot KIND --out FILE` renders a figure
or animation (needs `matplotlib`), with kinds `porter-error`, `porter-sign`,
`second-moment`, `variance-error` (single frames, `--at N`) and `1d-dist`,
`2d-dist` (animated GIFs).

---

For the mathematics behind any of this, see the
[book](https://tmfreiberg.github.io/euclids_algorithm_analysis/).