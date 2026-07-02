# Rust Froehlich-polaron DiagMC (rmc-frohlich)
#
#   make run                 run the release build against ./input.json
#
# Performance targets (all use release-tuned + target-cpu=native, pinned to core 0):
#   make test-perf-frohlich  end-to-end polaron sim in runs/test-perf-froehlich/alpha-5/
#   make test-perf-minimal   framework (bare) + physics (full) step-rate measurement
#   make perf                run both perf tests in sequence
#
# Criterion micro-benchmarks (rmc-core hot path):
#   make bench               run benchmarks, no baseline
#   make bench-before        run benchmarks and save results as the "before" baseline
#   make bench-after         run benchmarks and compare against the "before" baseline

SHELL := bash

CRATE              := rmc-frohlich
PERF_DIR           := runs/test-perf-froehlich/alpha-5
TUNED_BIN          := $(CURDIR)/target/release-tuned/$(CRATE)

MINIMAL_CRATE      := rmc-minimal
MINIMAL_TUNED_BIN  := $(CURDIR)/target/release-tuned/$(MINIMAL_CRATE)
MINIMAL_STEPS      ?= 100000000
MINIMAL_WARMUP     ?= 1000000

# Flags shared by all tuned builds and benchmarks.
TUNED_FLAGS        := RUSTFLAGS="-C target-cpu=native"

.DEFAULT_GOAL := run
.PHONY: run test-perf-frohlich test-perf-minimal perf bench bench-before bench-after

# Default: run the release build using input.json in the current directory.
# Results are written to ./results (the binary's default output directory).
run:
	cargo run --release -p $(CRATE) -- input.json

# End-to-end polaron simulation: build the fully-optimized "tuned" profile (fat LTO +
# codegen-units=1 + target-cpu=native), then cd into the perf folder and run it there so it
# reads that folder's input.json and writes every output file (summary, self-energy, ...) there.
# Pinned to core 0 and wrapped in `time` for a reproducible wall-clock number.
test-perf-frohlich:
	$(TUNED_FLAGS) cargo build --profile release-tuned -p $(CRATE)
	cd $(PERF_DIR) && time taskset -c 0 $(TUNED_BIN) input.json .

# Minimal step-rate harness: measures raw steps/sec for the framework hot path.
# Runs "bare" (no physics, pure framework overhead) then "full" (framework + minimal physics).
# Override step counts: make test-perf-minimal MINIMAL_STEPS=50000000 MINIMAL_WARMUP=100000
test-perf-minimal:
	$(TUNED_FLAGS) cargo build --profile release-tuned -p $(MINIMAL_CRATE)
	@echo "=== bare (framework overhead only) ==="
	time taskset -c 0 $(MINIMAL_TUNED_BIN) bare $(MINIMAL_STEPS) $(MINIMAL_WARMUP)
	@echo "=== full (framework + physics) ==="
	time taskset -c 0 $(MINIMAL_TUNED_BIN) full $(MINIMAL_STEPS) $(MINIMAL_WARMUP)

# Run both perf tests back-to-back for a comprehensive check before/after a change.
perf: test-perf-minimal test-perf-frohlich

# Criterion micro-benchmarks of rmc-core dispatch (static vs dyn, 10k-step loops).
# Uses target-cpu=native so baseline and comparison runs are built with identical flags.
bench:
	$(TUNED_FLAGS) taskset -c 0 cargo bench -p rmc-core

# Save current benchmark results as the "before" baseline, then make your change and run bench-after.
bench-before:
	$(TUNED_FLAGS) taskset -c 0 cargo bench -p rmc-core --bench hot_path -- --save-baseline before

# Compare current benchmark results against the "before" baseline saved by bench-before.
bench-after:
	$(TUNED_FLAGS) taskset -c 0 cargo bench -p rmc-core --bench hot_path -- --baseline before

bench-compare:
	cargo bench-compare -p rmc-minimal --bin rmc-minimal --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' --progress-regex 'step (\d+)/(\d+)' -- full 20000000
