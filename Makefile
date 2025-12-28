# Makefile for bambooML
SHELL = /bin/bash

.PHONY: style
style:
	black .
	flake8 --max-line-length=100 --extend-ignore=E203
	python3 -m isort .

.PHONY: test
test:
	python3 -m pytest tests/ --verbose --disable-warnings

.PHONY: test-cov
test-cov:
	python3 -m pytest tests/ --cov=bambooml --cov-report=html --cov-report=term --disable-warnings

.PHONY: clean
clean: style
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf .coverage* htmlcov/

.PHONY: install-dev
install-dev:
	pip install -e ".[dev]"
	pre-commit install

