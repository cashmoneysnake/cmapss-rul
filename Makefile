.PHONY: setup data reproduce reproduce-fast test serve docker-build docker-run

PY ?= python

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

data:
	$(PY) scripts/download_data.py

# Full reproduction: every reported run (3 seeds, both subsets, test set scored once each)
# into a FRESH MLflow store under reproduce/ (never your main mlflow.db), then the results table.
# Takes a few hours on a laptop CPU.
REPRO_DB = sqlite:///$(CURDIR)/reproduce/mlflow.db

reproduce: data
	rm -rf reproduce && mkdir -p reproduce
	for subset in FD001 FD004; do for model in xgboost lstm gru cnn transformer; do for seed in 0 1 2; do \
	  MLFLOW_TRACKING_URI=$(REPRO_DB) $(PY) -m src.train --subset $$subset --model $$model --seed $$seed --final || exit 1; \
	done; done; done
	for seed in 0 1 2; do \
	  MLFLOW_TRACKING_URI=$(REPRO_DB) $(PY) -m src.train --subset FD004 --model gru --window 70 --val-min-cycle 100 --seed $$seed --final || exit 1; \
	done
	MLFLOW_TRACKING_URI=$(REPRO_DB) $(PY) scripts/make_results_table.py --out reproduce/results.md

# Reduced mode used by CI and demos: FD001 only, one seed, tiny models (a couple of minutes).
FAST_DB = sqlite:///$(CURDIR)/reproduce_fast/mlflow.db

reproduce-fast: data
	rm -rf reproduce_fast && mkdir -p reproduce_fast
	MLFLOW_TRACKING_URI=$(FAST_DB) $(PY) -m src.train --subset FD001 --model xgboost --seed 0 --n-estimators 50 --final
	MLFLOW_TRACKING_URI=$(FAST_DB) $(PY) -m src.train --subset FD001 --model gru --seed 0 --epochs 2 --final
	MLFLOW_TRACKING_URI=$(FAST_DB) $(PY) scripts/make_results_table.py --out reproduce_fast/results.md

test:
	$(PY) -m pytest -q

serve:
	$(PY) -m uvicorn src.api.main:app --reload

docker-build:
	docker build -t cmapss-rul .

docker-run:
	docker run --rm -p 8000:8000 cmapss-rul
