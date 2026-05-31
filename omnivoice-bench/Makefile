# OmniVoice H100 bench — one-command driver.
# Usage:
#   make install       # set up venv + install deps (on the pod)
#   make corpus        # build prompts.jsonl + per-lang ref WAVs + speaker cache
#   make smoke         # tiny smoke sweep (naive, num_step=16, conc=1,4)
#   make sweep         # the full 120-cell sweep
#   make quality       # WER spot-check (num_step in {8,16,32})
#   make report        # render results/report.md from results/raw/
#   make all           # corpus -> sweep -> quality -> report

PY ?= python
RESULTS ?= results/raw

.PHONY: install corpus smoke sweep quality report all clean env

install:
	pip install --upgrade pip
	pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 \
		--extra-index-url https://download.pytorch.org/whl/cu128
	pip install -r requirements.txt

env:
	@echo "--- env ---" | tee results/env.txt
	@nvidia-smi | tee -a results/env.txt
	@$(PY) -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)" | tee -a results/env.txt
	@pip freeze | tee -a results/env.txt

corpus:
	$(PY) -m corpus.build_corpus

smoke:
	mkdir -p $(RESULTS)
	$(PY) -m bench.run_sweep --smoke --results-dir $(RESULTS)

sweep: env
	mkdir -p $(RESULTS)
	$(PY) -m bench.run_sweep --results-dir $(RESULTS)

quality:
	mkdir -p results/quality_samples
	$(PY) -m bench.quality_check

report:
	$(PY) -m bench.analyze --results-dir $(RESULTS)

all: corpus sweep quality report

clean:
	rm -rf $(RESULTS)/*.csv $(RESULTS)/*.log $(RESULTS)/*.json
