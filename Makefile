.PHONY: install lint typecheck test format check ingest ingest-dry eval serve

install:
	pip install -r requirements.txt -r requirements-dev.txt

ingest:
	python -m ingest.pipeline

ingest-dry:
	python -m ingest.pipeline --dry-run

# ARGS passes flags through, e.g. `make eval ARGS="--retrieval-only --k 8"`.
eval:
	python -m evals.run_eval $(ARGS)

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

serve:
	uvicorn app.main:app --reload --port 8000
