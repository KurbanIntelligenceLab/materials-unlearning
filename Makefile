PYTHON ?= python3
RELEASE_FLAGS ?=

.PHONY: release-precheck release-cli-help release-cleanup release-sanity release-lint release-build test

release-precheck: release-lint release-cli-help

release-cleanup:
	$(PYTHON) tools/release.py cleanup

release-cli-help:
	PYTHONPATH=src $(PYTHON) -m materials_unlearning.cli.main --help

release-lint:
	$(PYTHON) tools/release.py check $(RELEASE_FLAGS)

release-sanity: release-lint

test:
	PYTHONPATH=src $(PYTHON) -m unittest discover -s tests -v

release-build: release-precheck test
	$(PYTHON) tools/build_release.py
