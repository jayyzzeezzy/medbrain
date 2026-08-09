.PHONY: install lint typecheck test format check ingest ingest-dry

install:
	pip install -r requirements.txt -r requirements-dev.txt

ingest:
	python -m ingest.pipeline

ingest-dry:
	python -m ingest.pipeline --dry-run

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy .

test:
	pytest

format:
	ruff format .

check: lint typecheck test
