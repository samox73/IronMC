# Agent Instructions

## Repo Map

- `crates/rmc-core` - engine: updates, kernels, update sets, runner, seeding.
- `crates/rmc-{stats,grids,numeric,io}` - opt-in batteries (re-exported by the facade).
- `crates/rmc-diagmc` - imaginary-time diagrammatic MC engine behind a `Model` trait (not in the facade).
- `crates/rmc` - facade crate, feature-gated re-exports; `crates/rmc/examples/` are the API tutorials.
- `crates/apps/` - example applications built on the framework; they double as perf-regression fixtures. See `crates/apps/README.md`.
- `fixtures/` - inputs for the benchmark fixtures.

## Documentation scope

- Top-level docs (`README.md`, this file) describe the framework only.
- Each app is documented in its own `crates/apps/<app>/README.md`, with `crates/apps/README.md` as the index. Top-level docs may link to them but must not describe app physics or workflows.

## Build & Test

- Never run the full workspace test suite yourself. When full-suite verification is needed, give the user the exact `rtk cargo test --release --workspace` command and wait for their results.
- The debug profile is off limits by default. Run Cargo builds, checks, tests, and applications with `--release` (or `--profile release-tuned` for tuned runs). Use debug only after a concrete problem requires it, and state that reason first.
- Run only the smallest relevant release-mode test/check target yourself.
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

Benchmark commands may require `sudo` to isolate a CPU core. When benchmark verification is needed, tell the user exactly which command to run and wait for their results.

Copy-pasteable benchmark commands from `README.md`:

```bash
cargo bench-compare -p rmc-minimal --bin rmc-minimal --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' -- full 100000000
cargo bench-compare -p rmc-minimal --bin rmc-minimal --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' --rev-base HEAD -- full 100000000
cargo bench-compare -p rmc-frohlich --bin rmc-frohlich --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' -- bench fixtures/bench-frohlich.json
```

## Conventions

- Plan files live in `plans/{open,active,done}/*.md` (gitignored, local), sorted by status: `open` = not started, `active` = partly done, `done` = implemented.
- Do not put plan checkpoint names (A1, B3, etc.) anywhere else. Plan files are ephemeral and may not exist at a later point.
- Prefer existing `type: summary` commit subjects such as `feat:`, `fix:`, `chore:`, `refactor:`, and `docs:`.
- MC reproducibility is load-bearing: per-chain RNG streams derive from the master seed and a stable `ChainId`, never from scheduling order. Preserve this when touching the runner, seeding, or parallelism.
