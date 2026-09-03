# Kitsune — common entry points. GPU targets bill Modal; each prints its estimate and checks the
# budget guard (src/cost.py) before launching. Set the account with MODAL_PROFILE.
#   make test lint           CPU only, free
#   make data                 generate + filter + split the dataset (Modal, ~$6-10)
#   make train                pilot then main SFT (Modal)
#   make eval                 rebuild every table/figure from committed raw generations (CPU, free)
#   make demo                 run the Gradio demo locally (CPU, free)

PY      ?= uv run python
MODAL   ?= uv run modal
APP     := src/modal_app.py
PROFILE ?= kitsune30

.PHONY: help install test test-offline lint typecheck fmt data seeds train pilot main dpo merge gen-eval judge eval readme demo budget clean

help:
	@grep -E '^#' Makefile | sed 's/^# \{0,1\}//'

install:
	uv sync

test:
	uv run pytest -q

test-offline:
	uv run pytest -q -m "not network"

lint:
	uv run ruff check src tests demo
	uv run ruff format --check src tests demo

typecheck:
	uv run mypy

fmt:
	uv run ruff format src tests demo
	uv run ruff check --fix src tests demo

seeds:
	$(PY) -m kitsune.data.cli freeze-test

data: seeds
	MODAL_PROFILE=$(PROFILE) $(MODAL) run $(APP)::data_probe
	MODAL_PROFILE=$(PROFILE) $(MODAL) run $(APP)::data_generate
	MODAL_PROFILE=$(PROFILE) $(MODAL) run $(APP)::data_pull
	$(PY) -m kitsune.data.pipeline build

pilot:
	MODAL_PROFILE=$(PROFILE) $(MODAL) run $(APP)::train --config configs/train_pilot.yaml --name sft-pilot

main:
	MODAL_PROFILE=$(PROFILE) $(MODAL) run $(APP)::train --config configs/train_main.yaml --name sft-main

train: pilot main

dpo:
	MODAL_PROFILE=$(PROFILE) $(MODAL) run $(APP)::dpo --name dpo-main

merge:
	MODAL_PROFILE=$(PROFILE) $(MODAL) run $(APP)::merge --adapter dpo-main

gen-eval:
	MODAL_PROFILE=$(PROFILE) $(MODAL) run $(APP)::eval_generate

judge:
	MODAL_PROFILE=$(PROFILE) $(MODAL) run $(APP)::eval_judge

# Rebuilds reports/results.json, reports/results_table.md and reports/figures/ from raw generations.
eval:
	$(PY) -m kitsune.eval.report
	$(PY) -m kitsune.readme

readme:
	$(PY) -m kitsune.readme

demo:
	uv run --with-requirements demo/requirements.txt python demo/app.py

budget:
	$(PY) -m kitsune.cost render

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache reports/scratch
