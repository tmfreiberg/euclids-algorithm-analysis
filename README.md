# Euclid's Algorithm Analysis

[![Tests](https://github.com/tmfreiberg/euclids_algorithm_analysis/actions/workflows/tests.yml/badge.svg)](https://github.com/tmfreiberg/euclids_algorithm_analysis/actions/workflows/tests.yml)
[![Book](https://github.com/tmfreiberg/euclids_algorithm_analysis/actions/workflows/publish.yml/badge.svg)](https://github.com/tmfreiberg/euclids_algorithm_analysis/actions/workflows/publish.yml)

How many division steps does Euclid's algorithm take, on average, and how are
those step counts distributed? This project implements Euclid's algorithm
several ways, tabulates the step count `T(a, b)` over billions of integer pairs,
and compares the empirical distributions against the asymptotic (Gaussian)
behaviour predicted by the theorems of Heilbronn–Porter, Norton, Hensley, and
Baladi–Vallée — including an empirical estimate of the otherwise-unknown
subdominant constant `κ ≈ −0.1` in the variance.

📖 **The full exposition — mathematics, code, plots, and animations — is published as a Quarto book:**
**<https://tmfreiberg.github.io/euclids_algorithm_analysis/>**

This repository holds the reusable Python package (`euclid_analysis`), its test
suite, the precomputed data and figures, and the sources for that book.

Install it with **pip** — `pip install -e .` for the library, `-e ".[dev]"` to
add the test and lint tools (pytest, ruff, mypy), or `-e ".[dev,docs]"` to also
pull the Jupyter stack that builds the book. A **conda** `environment.yml` is
also provided (and bundles Quarto). See [Installation](#installation) below.

## What's here

```
book/                  Quarto book sources (.qmd) -> GitHub Pages
data/                  precomputed frequency tables (CSV)
docs/                  additional guides (e.g. the CLI reference)
images/                precomputed plots and animations (PNG/GIF)
src/euclid_analysis/   the installable package (see "The package" below)
tests/                 pytest suite, including validation against the data
```

## The package

```python
import euclid_analysis as ea

ea.gcd(1011, 69)            # 3
ea.count_steps(1011, 69)   # (3, 5)  -> gcd and step count T(a, b)
print(ea.write_euclid(1011, 69))   # a full aligned transcript

# Load a precomputed frequency table and summarise a column
C = ea.load_dataset("C")            # step counts over 0 < b < a < N, all pairs
stats = ea.dictionary_statistics(C[1001])
stats["mean"], stats["var"]         # ~5.40, ~3.41
```

Modules: `algorithms`, `fibonacci`, `constants`, `frequencies`, `statistics`,
`dataio`, `analysis`, and `plotting`. See the book for the full walk-through.

## Installation

### With pip (in a virtual environment)

```bash
python -m venv .venv
# Windows (cmd):        .venv\Scripts\activate
# Windows (PowerShell): .venv\Scripts\Activate.ps1
# macOS/Linux:          source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e ".[dev,docs]"
```

Pick the extras you need:

| Command | Gets you |
| --- | --- |
| `pip install -e .` | the `euclid_analysis` library only |
| `pip install -e ".[dev]"` | library **+ dev tools** (pytest, pytest-cov, ruff, mypy) |
| `pip install -e ".[docs]"` | library **+ book toolchain** (Jupyter, ipykernel, pyyaml) |
| `pip install -e ".[dev,docs]"` | everything, for running tests *and* building the book |

Requires Python ≥ 3.10; the scientific stack (`numpy`, `pandas`, `scipy`,
`matplotlib`) is installed automatically.

### With conda

```bash
conda env create -f environment.yml
conda activate euclid-analysis
```

The conda environment also bundles Quarto, so no separate install is needed.

## The package in more depth

The library is organised into focused modules:

- **`algorithms`** — `gcd`, `gcdn`, `count_steps`, `remainders`,
  `quotients_remainders`, and `write_euclid` (an aligned transcript).
- **`fibonacci`** — `fib` (O(n) memoised), `fibonacci_list` (tabulation),
  `naive_fib`, and `binet_fib`; consecutive Fibonacci numbers are the worst case.
- **`constants`** — the mean/variance constants (`LAMBDA_DIXON`,
  `PORTER_CONSTANT`, `NU_NORTON`, `ETA_HENSLEY`, `KAPPA_VAR`, …).
- **`frequencies`** — `heilbronn` (one-dimensional) and `euclid_alg_frequencies`
  (two-dimensional) build the step-count frequency tables.
- **`statistics`** — `dictionary_statistics`, `basic_stats`, `dists`, `tabulate`.
- **`dataio`** — `load_dataset` / `save_meta_dict` / `load_meta_dict` round-trip
  the tables to CSV.
- **`analysis`** — the numerical investigation: Porter error terms, error
  champions, sign-sum walk, and the curve fits that estimate the subdominant
  constants.
- **`plotting`** — figure and animation builders (require matplotlib).

## Command-line interface

Installing the package also provides a terminal tool — `euclid-analysis`, or the
short alias `ea` — that exposes most of the library without writing any Python:

```bash
ea euclid 1011 69      # a transcript of Euclid's algorithm
ea constants           # the golden ratio and the mean/variance constants
ea fit-constants       # the subdominant constants from the shipped data
ea --help              # the full list of commands
```

See [`docs/cli.md`](docs/cli.md) for the full guide.

## Data

The frequency tables in `data/` record, for each checkpoint `N` (or each `a` in
the one-dimensional case), how many pairs achieve each step count. The largest,
`gcd_pairs_100001df.csv` (~11 MB), is included but not required: the book and
tests run without it. To regenerate any table from scratch:

```python
_, B, C = ea.euclid_alg_frequencies([1], [], list(range(1, 101001, 1000)), {}, {}, {})
```

(The full run over pairs up to 100 000 takes several hours — hence the
precomputed CSVs.)

## Testing

```bash
pytest                 # full suite
pytest -m "not slow"   # skip the data-regeneration checks
```

The `slow`-marked tests regenerate small portions of the published tables and
assert they match the shipped CSVs cell for cell.

## Building the book locally

The book renders its fast code cells live via Jupyter (installed by the `docs`
extra above). [Quarto](https://quarto.org/docs/get-started/) must be on your
PATH — the conda environment includes it; pip users install it separately.

```bash
quarto preview    # live preview at the repo root, or: quarto render
```

If Quarto can't find your virtual environment's Python (common on Windows), set
`QUARTO_PYTHON` to the venv interpreter first:

```bash
# Windows (cmd)
set QUARTO_PYTHON=%CD%\.venv\Scripts\python.exe
# PowerShell
$env:QUARTO_PYTHON = "$PWD\.venv\Scripts\python.exe"
# macOS/Linux
export QUARTO_PYTHON="$PWD/.venv/bin/python"
```

Run `quarto check jupyter` to confirm it sees the venv and a `python3` kernel.

## License

MIT. See [`LICENSE`](LICENSE).