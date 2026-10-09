PY ?= python

.PHONY: all data features experiment dashboard test

all: experiment

data:            ## download + extract the raw IMS dataset (~1 GB download, ~6 GB on disk)
	$(PY) scripts/download_data.py --out data/raw

features:        ## raw snapshots -> data/features/*.csv.gz (already committed)
	$(PY) scripts/build_features.py --raw data/raw --out data/features

experiment:      ## fit detectors, raise alarms, write reports/ (runs from the committed features)
	$(PY) scripts/run_experiment.py --config configs/default.yaml

dashboard:       ## interactive replay of a test (http://localhost:8501)
	streamlit run app/dashboard.py

test:
	$(PY) -m pytest -q
