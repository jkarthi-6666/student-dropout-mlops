.PHONY: setup stack train infer drift test lint clean

setup:
	uv venv --python 3.12
	uv pip install -e ".[dev]"

stack:
	uv run zenml init
	uv run zenml experiment-tracker register mlflow_tracker --flavor=mlflow
	uv run zenml model-registry register mlflow_registry --flavor=mlflow
	uv run zenml stack register dropout_stack \
		-a default -o default \
		-e mlflow_tracker -r mlflow_registry --set
	uv run zenml stack describe

train:
	uv run python -m dropout_risk.pipelines.training

infer:
	uv run python -m dropout_risk.pipelines.inference

drift:
	uv run python -m dropout_risk.pipelines.drift

test:
	uv run pytest

lint:
	uv run ruff check src tests

ui:
	uv run mlflow ui --port 5000

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
