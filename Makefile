PYTHON ?= python3
JOBS ?= 4
BUILD_DIR ?= hr_edl/build.optimized

.PHONY: build test verify figures tables smoke main ablation
build:
	cmake -S hr_edl/src -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel $(JOBS)

test:
	$(PYTHON) -m unittest discover -s tests -v
	ctest --test-dir $(BUILD_DIR) --output-on-failure

verify:
	$(PYTHON) scripts/verify_release.py --full

figures:
	$(PYTHON) bin/plot_SCD.py --main-data data/raw/main --ablation-data data/raw/ablation --output-dir outputs/figures

tables:
	$(PYTHON) scripts/generate_tables.py --output outputs/tables

smoke:
	$(PYTHON) bin/run_SCD_main.py --games kuhn_3p --iterations 3 --exe-dir $(BUILD_DIR)/bin --data-dir outputs/smoke/main
	$(PYTHON) bin/run_SCD_ablation.py --games kuhn_3p --iterations 3 --exe-dir $(BUILD_DIR)/bin --data-dir outputs/smoke/ablation

main:
	$(PYTHON) bin/run_SCD_main.py --exe-dir $(BUILD_DIR)/bin --threads-per-run 4 --max-parallel-jobs 8

ablation:
	$(PYTHON) bin/run_SCD_ablation.py --exe-dir $(BUILD_DIR)/bin --threads-per-run 2 --max-parallel-jobs 6
