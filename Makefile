.PHONY: setup data reproduce reproduce-fast test serve docker-build docker-run

PY ?= python

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

# Placeholders until train.py / evaluate.py exist (Week 1)
reproduce:
	@echo "TODO: full train + evaluate on FD001-FD004, all runs logged to MLflow"

reproduce-fast:
	@echo "TODO: tiny end-to-end run used by CI"

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m uvicorn src.api.main:app --reload

docker-build:
	docker build -t cmapss-rul .

docker-run:
	docker run --rm -p 8000:8000 cmapss-rul
