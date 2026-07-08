# Agent Instructions

## Repo Map

- `crates/rmc-core` - engine: updates, kernels, update sets, runner, seeding.
- `crates/rmc-{stats,grids,numeric,io}` - opt-in batteries.
- `crates/rmc` - facade crate, feature-gated re-exports.
- `crates/apps/rmc-minimal` - hot-path benchmark fixture.
- `crates/apps/rmc-frohlich` - full polaron engine and perf fixture.

## Build & Test

- `rtk cargo test --workspace` runs the workspace test suite.
- `make run` does the default Frohlich-polaron run.
- The `release-tuned` profile exists for tuned performance builds; see `README.md`.

## Shell

Prefix shell commands with `rtk`.

Examples:

```bash
rtk git status
rtk cargo test
```

## Benchmarks

Do not run benchmark commands yourself.

Benchmark commands may require `sudo` to isolate a CPU core. When benchmark verification is needed,
tell the user exactly which command to run and wait for their results.

Copy-pasteable benchmark commands from `README.md`:

```bash
cargo bench-compare -p rmc-minimal --bin rmc-minimal --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' -- full 100000000
cargo bench-compare -p rmc-minimal --bin rmc-minimal --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' --rev-base HEAD -- full 100000000
cargo bench-compare -p rmc-frohlich --bin rmc-frohlich --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' -- bench fixtures/bench-frohlich.json
```

## Conventions

- Plan files live at the repo root as `plan-*.md`.
- Prefer existing `type: summary` commit subjects such as `chore:`, `refactor:`, and `fix:`.
