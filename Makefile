# this file is to generate the documentation site locally for testing

.PHONY: docs docs-clean

docs:
	cd docs && poetry run sphinx-build -b html . _build/html


docs-clean:
	rm -rf docs/_build
