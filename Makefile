# IronMC framework + example apps.
#
#   make run        run the Fröhlich-polaron example against crates/apps/rmc-frohlich/input.json
#
# Benchmarks (cargo bench-compare; run them yourself, see AGENTS.md):
#   make bench           all three below
#   make bench-core      criterion hot-path micro-benchmarks (rmc-core)
#   make bench-minimal   rmc-minimal step rate (bare framework, then full physics)
#   make bench-frohlich  rmc-frohlich end-to-end step rate

# Recipes are nushell. Make expands $(VAR) before nu sees the line, so any $
# meant for nushell (variables, $"...") must be written $$ to survive make.
# .ONESHELL runs each recipe as one nu script, so `let` bindings persist across
# lines; --no-config-file skips loading the user's env.nu/config.nu.
SHELL := nu
.SHELLFLAGS := --no-config-file -c
.ONESHELL:

CRATE := rmc-frohlich

.DEFAULT_GOAL := run

.PHONY: run bench bench-core bench-minimal bench-frohlich

# Release build against the example's committed input.json; results go to ./results.
run:
	cargo run --release -p $(CRATE) -- crates/apps/$(CRATE)/input.json

bench: bench-core bench-minimal bench-frohlich

bench-core:
	@cargo bench-compare --bench hot_path --dedicate-core
	print "------------------------------------------------------"

bench-minimal:
	@cargo bench-compare --bin rmc-minimal --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --progress-regex 'step (\d+)/(\d+)' --dedicate-core -- bare 3000000
	print "------------------------------------------------------"
	cargo bench-compare --bin rmc-minimal --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --progress-regex 'step (\d+)/(\d+)' --dedicate-core -- full 3000000
	print "------------------------------------------------------"

bench-frohlich:
	@cargo bench-compare --bin rmc-frohlich --reps 10 --metric-regex 'steps/sec:\s*([\d.]+)' --dedicate-core -- bench
	print "------------------------------------------------------"
