.PHONY: install lint typecheck test format check

install:
	pip install -r requirements.txt -r requirements-dev.txt

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
