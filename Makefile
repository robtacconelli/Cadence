.PHONY: help data headline lossless codec figures registry check verify
help:
	@echo "make data      - fetch grid+transit (headline corpora)"
	@echo "make headline  - reproduce the +21.4% demand result (needs GPU)"
	@echo "make codec     - bit-exact round trip (needs GPU)"
	@echo "make lossless  - the +0.03% null result (uses committed cache)"
	@echo "make figures   - regenerate paper figures from results/"
	@echo "make check     - compile-check every script"
	@echo "make verify    - check every paper number against results/ (no GPU)"
data:
	python scripts/fetch_data.py grid transit
headline:
	python experiments/grid_bench.py gpu && python experiments/grid_bench.py
	python experiments/transit_bench.py gpu && python experiments/transit_bench.py
	python experiments/rescore_all.py
codec:
	python experiments/codec_demo.py
lossless:
	python experiments/fair_compare.py
figures:
	python scripts/make_figures.py
registry:
	python experiments/registry.py
verify:
	python scripts/verify_paper.py
check:
	@for f in cli.py experiments/*.py cadence/*.py scripts/*.py; do \
		python -m py_compile $$f || exit 1; done; echo "all scripts compile"
