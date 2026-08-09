.PHONY: install lint typecheck test format check ingest ingest-dry eval

install:
	pip install -r requirements.txt -r requirements-dev.txt

ingest:
	python -m ingest.pipeline

ingest-dry:
	python -m ingest.pipeline --dry-run

eval:
	python -m evals.run_eval

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
