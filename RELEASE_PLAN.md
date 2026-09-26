# Release checks

1. Run commands from the repository root with Python 3.10 or newer.
2. Run `make release-precheck` to check release files, path hygiene, available evidence checksums, and the CLI. This target does not move files.
3. For `anon`, also run `make release-lint RELEASE_FLAGS=--anonymous`. Use `python tools/release.py check --anonymous --history` to check author/committer metadata and historical licenses. Review all historical file contents and the hosting identity separately.
4. Keep `data/`, `development/`, `.lake/`, environments, and generated outputs out of version control and distributions.
5. Run the paired saved-evidence analysis and the proof audit described in `proofs/README.md`.
6. Install the release extra and use `make release-build PYTHON=python` to remove archive owner metadata and timestamps. Inspect both archives and repeat installation and smoke checks from a clean checkout.
7. Use `make release-cleanup` only when intentionally archiving private intermediate artifacts under `development/development_core/retired_results`.
8. Verify branch license holders and review the complete staged file list before committing. Publish only the intended branches to the intended destination.
