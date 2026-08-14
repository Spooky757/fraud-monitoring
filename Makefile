# Thin wrappers around the CLI. Every target here is exactly what CI runs.
.PHONY: install test lint reference monitor demo simulate clean

PY := PYTHONPATH=src python
ENV ?= dev

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=src pytest tests/ -v

lint:
	python -m pyflakes src/ tests/

# Freeze the reference distribution for the current champion. Re-run this every
# time a new model is promoted, or drift will be measured against the wrong past.
reference:
	$(PY) -m fraud_monitoring.cli --env $(ENV) build-reference \
		--training-data data/raw/creditcard.csv

# Score one batch and decide. BATCH=path/to/file.csv
monitor:
	$(PY) -m fraud_monitoring.cli --env $(ENV) monitor --batch $(BATCH)

simulate:
	$(PY) -m fraud_monitoring.cli --env $(ENV) simulate \
		--out data/incoming/simulated.csv --rows 20000 --kind $(KIND) --magnitude $(MAG)

# End-to-end walkthrough on synthetic data: train a champion, build the reference,
# then watch a clean batch pass and drifted batches escalate.
demo:
	$(PY) scripts/demo.py

clean:
	rm -rf .pytest_cache __pycache__ .retrain-workdir artifacts/challenger artifacts/proposed
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
