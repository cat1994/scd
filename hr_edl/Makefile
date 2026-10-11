BUILD_DIR ?= build.optimized
JOBS ?= 4
.PHONY: all test
all:
	cmake -S src -B $(BUILD_DIR) -DCMAKE_BUILD_TYPE=Release
	cmake --build $(BUILD_DIR) --parallel $(JOBS)
test: all
	ctest --test-dir $(BUILD_DIR) --output-on-failure
