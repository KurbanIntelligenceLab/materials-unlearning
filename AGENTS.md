# Agent guide

## Scope and working rules

This repository contains reference implementations, recorded numerical evidence, and Lean proofs for the paper described in `README.md`. Read the README before changing scientific behavior, and inspect the current branch and working tree before editing.

- Keep changes focused on the requested task. Preserve existing user changes.
- Use Markdown for documentation. Keep public documentation about installation, methods, and reproducibility; omit private research workflows, local machine details, and internal notes.
- Treat ignored local files as private. Do not stage, package, upload, or remove them as part of routine cleanup. Use an explicit file list when staging.
- Preserve the user's configured Git identity. Do not add assistant names or co-author trailers. Do not rewrite history or push other branches as a side effect of a requested change.
- Verify claims against source code and recorded outputs. For external API or dependency behavior, use the relevant upstream documentation. Distinguish a successful test from a full experiment reproduction.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/materials_unlearning/cli/main.py` | Console command dispatch |
| `src/materials_unlearning/cli/shared.py` | Shared arguments, data loading, splitting, and backend setup |
| `src/materials_unlearning/configuration.py` | Repository-relative resource paths and ridge configuration |
| `src/materials_unlearning/core.py` | Corpus handling, descriptors, deletion measurements, and statistical utilities |
| `src/materials_unlearning/backends.py` | Fixed-feature ridge and CGCNN implementations |
| `src/materials_unlearning/unlearning.py` | Retraining reference and approximate update methods |
| `src/materials_unlearning/paired.py` | Replay and consistency checks using saved ridge evidence |
| `results/` | Recorded evidence; treat as immutable unless explicitly updating results |
| `tests/` | Regression tests for numerical behavior, data alignment, and release metadata |
| `proofs/` | Pinned Lean project and theorem audit |
| `tools/` | Release checks and archive construction |

Run Python commands from the repository root. Resource paths are resolved relative to the working directory. Imports must not launch experiments or write output. Keep `_cli.py` as a compatibility shim; shared CLI behavior belongs in `cli/`.

## Setup

Python 3.10 or newer is required. The base package needs NumPy and SciPy.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

Install optional dependencies only for the relevant task:

```sh
python -m pip install -e '.[materials]'  # structure processing and Materials Project access
python -m pip install -e '.[neural]'     # graph-model training
python -m pip install -e '.[release]'    # distribution builds
```

Inspect command arguments with `materials-unlearning --help` or `python -m materials_unlearning.<module> --help`.

## Fast verification without datasets

```sh
make test PYTHON=python
make release-precheck PYTHON=python
python -m materials_unlearning.paired --out runs/paired
```

The paired command verifies saved request-level identities and numerical relationships and writes `runs/paired/summary.json`. Compare it with `results/paired/summary.json`; investigate differences before claiming reproduction. Tests and the release precheck do not replace experiment reproduction or formal proof verification.

Always supply an explicit `--out runs/<analysis>` to analysis commands. Preserve `results/`; never use it as an output directory. Keep generated outputs ignored.

For a small synthetic integration check of approximate methods:

```sh
python -m materials_unlearning.operating_points --demo --n-features 16 --n-requests 3 --seeds 1 --out runs/demo_operating
```

This is a pipeline test only. Demo results and reduced settings must never be presented as the paper's results.

## Data-dependent reproduction

`data/mp20_full.npz` supplies descriptors and targets. CGCNN training additionally requires the aligned `data/mp20_graphs.npz`. `data/mp20_snapshot.json` records dataset provenance. Verify available inputs against `results/files.json` before claiming exact replay; a new database export is a new dataset.

Descriptor caches must use non-object arrays and load with pickling disabled. Graph caches contain pickled dictionaries and must come from a trusted source. Preserve record IDs and row order across descriptors, targets, and graphs. The cached-input `--limit` option selects the same leading rows from both caches.

Use `MP_API_KEY` in the environment for Materials Project access. Never put credentials in source, logs, documentation, or saved results.

A reduced ridge check with the descriptor cache is:

```sh
python -m materials_unlearning.ridge --features 16 --corpus-size 300 --diagnostic-requests 3 --out runs/ridge_smoke
```

For the paper's standard and high-capacity settings, use the commands in `README.md`. Full ridge experiments, controlled redundancy sweeps, and repeated neural fitting are substantial computations; use them when the task requires those experiments, rather than as routine installation checks. Neural fitting must explicitly select `--backend cgcnn` and provide `--graphs`; the shared CLI defaults to ridge.

## Scientific invariants

- Hold the original training split's normalization, bandwidth, and random-feature basis fixed for the ridge retraining reference. Preserve recorded seeds and request-selection rules.
- Pass the training partition to deletion measurements. Retained retraining must exclude both the deleted records and held-out test records.
- Keep reference target loss, prediction change, and retained utility distinct. A low target loss after deletion does not by itself establish unlearning failure.
- Distinct record identifiers can have duplicate features and labels. Removing one record must not silently remove its retained relatives.
- Saved neural analyses condition on one training seed. Request-bootstrap intervals do not measure training-seed uncertainty.
- Preserve the original `script_sha256` provenance fields. Do not replace them with refactored source hashes. Update file checksums only when intentionally replacing the corresponding evidence, and document why.
- Preserve saved evidence bytes, including line endings. `.gitattributes` protects result-file hashes across platforms.

For numerical changes, test against an independent reference such as an augmented least-squares solve or direct retained-data refit. Check that approximate updates leave the supplied original model unchanged. Do not silence numerical warnings to make a check pass; inspect finite outputs and reference discrepancies.

## Formal proofs

Read `proofs/README.md` for scope and the manuscript-to-theorem map. From `proofs/`, run:

```sh
lake exe cache get
python3 verify.py
```

Install the toolchain named in `lean-toolchain`. Preserve `lake-manifest.json`; do not run `lake update` during reproduction. If needed, set `LAKE` to the installed executable. The verifier builds project modules and checks theorem coverage and transitive axioms. Do not introduce proof placeholders or additional axioms to make the build pass. A successful Lean audit does not certify Python experiments.

## Releases and branches

`main` contains identified publication metadata. `anon` is intended for anonymous distribution. Do not copy author names, affiliations, contact details, or identifying metadata into an anonymous release. An anonymous license alone does not anonymize Git history or a hosting account.

Before committing, review `git diff --check` and the explicit staged file list. Inspect names, email addresses, machine paths, credentials, and unintended generated files. Keep license attribution appropriate to the target branch.

For an anonymous checkout, audit both current files and history:

```sh
python tools/release.py check --anonymous --history
```

A failed history audit is a publication blocker, not a reason to rewrite history without authorization. The checker covers selected patterns and metadata; separately inspect all historical file contents and the intended hosting destination.

Build release artifacts through the sanitized build target:

```sh
make release-build PYTHON=python
```

Inspect both the source archive and wheel, including archive ownership, timestamps, license contents, and unexpected files. The source archive includes evidence, proofs, tests, and release tools; the wheel contains the Python package and distribution metadata. Verify installation and smoke checks from a clean checkout or extracted source archive. Publish only the requested branch or verified artifacts; never publish an archive of the entire local working directory.

When reporting completion, state what changed, which checks ran, whether saved evidence matched, and any checks that were not run. Do not claim a full reproduction from help output, a demo, or a reduced experiment.
