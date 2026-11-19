# Iraqi Voice Studio — unified project control (Makefile)
# On Windows/Colab use the equivalent CLI: `uv run rvc-platform <command>`
# (every make target maps 1:1 to a CLI command; see `make help`)

PY ?= python
UV ?= uv
RUN = $(UV) run
CLI = $(RUN) rvc-platform
PORT ?= 8320
EXPERIMENT ?= iraqi_voice
DATASET ?= data/raw/iraqi_dataset

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------- install ----
.PHONY: install setup check doctor clean
install: ## Create the uv environment + install dependencies (all groups)
	$(UV) sync --all-groups
setup: install ## Alias of install (kept for muscle memory)
	$(UV) sync --all-groups
check: ## Verify dependency imports and versions
	$(CLI) install check
doctor: ## Deep environment audit (GPU/CUDA/assets); --fix via FIX=1
	$(CLI) install doctor $(if $(FIX),--fix,)
clean: ## Remove caches and temp files (use clean-all for runs+data)
	$(CLI) clean

.PHONY: clean-all
clean-all: ## Remove caches + runs + uploads + conversions
	$(CLI) clean --all

# ------------------------------------------------------------------ env ------
.PHONY: env sync update lock verify env-gpu
env: ## Show environment/paths overview
	$(CLI) env info
sync: ## uv sync --all-groups
	$(UV) sync --all-groups
update: ## Upgrade locked dependencies
	$(UV) sync --upgrade --all-groups
lock: ## Regenerate uv.lock
	$(UV) lock
verify: ## Reproducibility: sync from lockfile + import check
	$(UV) sync --frozen --all-groups && $(CLI) install check
env-gpu: ## Install CUDA torch into the venv (CUDA=cu128|cu126|cu118)
	$(CLI) env gpu $(CUDA)

# ---------------------------------------------------------------- assets ----
.PHONY: download-models data-check prepare-audio-dataset preprocess extract-features prepare-dataset
download-models: ## Download HuBERT + RMVPE + pretrained v2 bases
	$(CLI) download-models
data-check: ## Validate a dataset folder (quality/corruption/duplicates)
	$(CLI) data-check $(DATASET)
prepare-audio-dataset: ## Import/clean external audio into data/raw (SOURCE= NAME=)
	$(CLI) prepare-audio-dataset $(SOURCE) $(NAME) --sr $(SR) $(FORCE)
preprocess: ## Slice + normalise dataset into run chunks
	$(CLI) preprocess $(DATASET) $(EXPERIMENT) --sr $(SR)
extract-features: ## Extract F0 + HuBERT features for an existing run
	$(CLI) extract-features $(EXPERIMENT) --f0-method $(F0_METHOD)
prepare-dataset: ## Full data pipeline: check→preprocess→F0→features→index
	$(CLI) prepare-dataset $(DATASET) $(EXPERIMENT) --sr $(SR) --f0-method $(F0_METHOD)

# -------------------------------------------------------------- training ----
.PHONY: train train-resume train-stop checkpoints export
train: ## Start training (in-process, live progress; DATASET= EPOCHS= ...)
	$(CLI) train start --dataset $(DATASET) --experiment $(EXPERIMENT) \
	--epochs $(EPOCHS) --batch $(BATCH) --sr $(SR)
train-resume: ## Resume the latest (or EXPERIMENT=) run from its checkpoint
	$(CLI) train start --experiment $(EXPERIMENT)
train-stop: ## Cooperative stop of the trainer (saves checkpoint first)
	$(CLI) train stop
checkpoints: ## List checkpoints of a run
	$(CLI) train checkpoints $(EXPERIMENT)
export: ## Export a trained run into models/ (EXPERIMENT=, EPOCH= optional)
	$(CLI) train export $(EXPERIMENT) $(if $(EPOCH),--epoch $(EPOCH))

# ---------------------------------------------------------------- models ----
.PHONY: model-list model-load model-info inference
model-list: ## List available voice models
	$(CLI) model list
model-load: ## (API) load MODEL= into the engine of a running server
	curl -s -X POST http://127.0.0.1:$(PORT)/api/models/$(MODEL)/load | head -c 400; echo
model-info: ## Inspect MODEL= checkpoint metadata
	$(CLI) model info $(MODEL)
inference: ## Convert INPUT= audio with MODEL= (Audio-to-Audio)
	$(CLI) inference $(INPUT) $(MODEL) --pitch $(PITCH) --index-rate $(INDEX_RATE)

# ------------------------------------------------------------------- web ----
.PHONY: web web-stop web-status realtime realtime-stop realtime-status
web: ## Launch the web dashboard (blocks; --tunnel via TUNNEL=1)
ifeq ($(TUNNEL),1)
	$(CLI) web --tunnel --port $(PORT)
else
	$(CLI) web --port $(PORT)
endif
web-stop: ## Stop the web server
	$(CLI) web-stop
web-status: ## Show live platform status from the running server
	$(CLI) web-status

realtime: ## Latency benchmark of MODEL= with synthetic blocks (real engine)
	$(CLI) realtime-benchmark $(MODEL) --blocks $(BLOCKS) --block-ms $(BLOCK_MS)
realtime-stop: ## Real-time runs inside the web session; stopping web stops it
	$(CLI) web-stop
realtime-status: ## Latency stats of live realtime sessions
	@curl -s http://127.0.0.1:$(PORT)/api/realtime/stats | head -c 600; echo

# ---------------------------------------------------------------- tunnel ----
.PHONY: tunnel tunnel-stop tunnel-status
tunnel: ## Expose the running web server via a free Cloudflare quick tunnel
	@curl -s -X POST http://127.0.0.1:$(PORT)/api/tunnel/start | head -c 300; echo
tunnel-stop: ## Stop the tunnel
	@curl -s -X POST http://127.0.0.1:$(PORT)/api/tunnel/stop; echo
tunnel-status: ## Tunnel state + public URL
	@curl -s http://127.0.0.1:$(PORT)/api/tunnel/status; echo

# ------------------------------------------------------------- lifecycle ----
.PHONY: start stop restart status logs
start: web ## Start the platform (web server)
stop: ## Stop everything (web + tunnel + trainer)
	$(CLI) web-stop
restart: stop start ## Restart the platform
status: ## Platform status (live server or offline snapshot)
	$(CLI) status
logs: ## Tail platform logs (LINES=)
	$(CLI) logs --lines $(LINES)

# ------------------------------------------------------------------ tests ----
.PHONY: test test-unit test-integration
test: ## Run the full test suite
	$(UV) run pytest -q
test-unit: ## Unit tests only
	$(UV) run pytest tests/unit -q
test-integration: ## Integration tests only
	$(UV) run pytest tests/integration -q

# ---------------------------------------------------------------- defaults ---
SR ?= 40k
F0_METHOD ?= rmvpe
EPOCHS ?= 200
BATCH ?= 6
PITCH ?= 0
INDEX_RATE ?= 0
BLOCKS ?= 40
BLOCK_MS ?= 160
MODEL ?=
INPUT ?=
