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
.PHONY: run bench bench-compare-core bench-compare-minimal bench-compare-frohlich

# Default: run the release build using input.json in the current directory.
# Results are written to ./results (the binary's default output directory).
run:
	cargo run --release -p $(CRATE) -- input.json

bench: bench-compare-core bench-compare-minimal bench-compare-frohlich

bench-core:
	cargo bench-compare --bench hot_path --dedicate-core

bench-minimal:
	cargo bench-compare --bin rmc-minimal --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --progress-regex 'step (\d+)/(\d+)' --dedicate-core -- bare 3000000
	cargo bench-compare --bin rmc-minimal --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --progress-regex 'step (\d+)/(\d+)' --dedicate-core -- full 3000000

bench-frohlich:
	cargo bench-compare --bin rmc-frohlich --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --dedicate-core -- bench
