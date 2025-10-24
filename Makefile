.DEFAULT_GOAL := run

VENV := .venv
PY := $(VENV)/bin/python3
PIP := $(VENV)/bin/pip
UVICORN := $(VENV)/bin/uvicorn

# Service settings (override at call time, e.g. `make run PORT=8000`)
HOST ?= 127.0.0.1
PORT ?= 9000
APP  ?= service.app:app

# AlignScore runtime settings
ALIGNSCORE_MODEL_PATH ?= models/AlignScore-large.ckpt
ALIGNSCORE_DEVICE ?= auto
ALIGNSCORE_EVAL_MODE ?= nli_sp
ALIGNSCORE_BATCH_SIZE ?= 8

.PHONY: run dev install venv clean smoke spaCy docker-build docker-run

run: install
	@echo "Running AlignScore service at http://$(HOST):$(PORT)"
	@ALIGNSCORE_MODEL_PATH=$(ALIGNSCORE_MODEL_PATH) \
	ALIGNSCORE_DEVICE=$(ALIGNSCORE_DEVICE) \
	ALIGNSCORE_EVAL_MODE=$(ALIGNSCORE_EVAL_MODE) \
	ALIGNSCORE_BATCH_SIZE=$(ALIGNSCORE_BATCH_SIZE) \
	$(UVICORN) $(APP) --host $(HOST) --port $(PORT)

dev: install
	@echo "Running AlignScore service in reload mode at http://$(HOST):$(PORT)"
	@ALIGNSCORE_MODEL_PATH=$(ALIGNSCORE_MODEL_PATH) \
	ALIGNSCORE_DEVICE=$(ALIGNSCORE_DEVICE) \
	ALIGNSCORE_EVAL_MODE=$(ALIGNSCORE_EVAL_MODE) \
	ALIGNSCORE_BATCH_SIZE=$(ALIGNSCORE_BATCH_SIZE) \
	$(UVICORN) $(APP) --host $(HOST) --port $(PORT) --reload

smoke: install
	@echo "Running AlignScore smoke check"
	@ALIGNSCORE_MODEL_PATH=$(ALIGNSCORE_MODEL_PATH) $(PY) service/smoke_check.py

spaCy: install
	@echo "Installing spaCy English model (en_core_web_sm)"
	@$(PY) -m spacy download en_core_web_sm

install: venv
	@echo "Installing project and service dependencies into $(VENV)"
	@$(PIP) install -U pip setuptools wheel
	@# Core library (AlignScore) and its deps
	@$(PIP) install -e .
	@# Service runtime
	@$(PIP) install fastapi uvicorn pydantic starlette

venv:
	@if [ ! -d "$(VENV)" ]; then \
		python3 -m venv $(VENV); \
		. $(VENV)/bin/activate; \
		$(PIP) install -U pip; \
	fi

clean:
	rm -rf $(VENV) *.pyc __pycache__ */__pycache__

# Optional Docker helpers (build a CPU-only image by default)
docker-build:
	docker build -t alignscore-service:latest .

docker-run:
	docker run --rm -it \
	  -e ALIGNSCORE_MODEL_PATH=/models/AlignScore-large.ckpt \
	  -e ALIGNSCORE_DEVICE=$(ALIGNSCORE_DEVICE) \
	  -e ALIGNSCORE_EVAL_MODE=$(ALIGNSCORE_EVAL_MODE) \
	  -e ALIGNSCORE_BATCH_SIZE=$(ALIGNSCORE_BATCH_SIZE) \
	  -p $(PORT):9000 \
	  -v $$(pwd)/models:/models \
	  alignscore-service:latest

