# simplemc Rust workspace

This workspace is the Rust port of the `simplemc` Monte-Carlo framework. The core framework lives in `rmc-core`, with supporting numeric, grid, IO, and statistics crates plus physics/application crates such as `rmc-frohlich` and the minimal benchmark harness in `rmc-minimal`. Use `make run` for the default Frohlich-polaron run and `cargo test --workspace` for the workspace test suite.

## Performance testing

Install the comparison tool with:

```nu
cargo install --git https://github.com/samox73/cargo-bench-compare
```

Both benchmark binaries do a one-shot run and print a `steps/sec: <value>` line. Repetitions, revision checkout, the tuned profile (`release-tuned`), and `-C target-cpu=native` are handled by `cargo bench-compare`; use `--runs-on-core <n>` for CPU pinning instead of the manual `taskset` used by the Makefile targets.

The examples below are written as single logical commands so they work in Bash, Nushell,
and other common shells. Replace `BASE_SHA` with the base revision you want to compare
against.

```nu
# framework hot path (rmc-minimal), current state vs the merge-base
cargo bench-compare -p rmc-minimal --bin rmc-minimal --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' -- full 100000000

# framework hot path (rmc-minimal), current (unstaged) state vs the last commit
cargo bench-compare -p rmc-minimal --bin rmc-minimal --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' --rev-base HEAD -- full 100000000

# full polaron engine (rmc-frohlich)
cargo bench-compare -p rmc-frohlich --bin rmc-frohlich --reps 5 --metric-regex 'steps/sec:\s*([\d.]+)' -- bench fixtures/bench-frohlich.json
```

Caveat: the `steps/sec:` line only exists from the commit that introduced this performance harness format onward. When comparing against older revisions, prefer comparing only post-change commits; otherwise drop `--metric-regex` to use wall-clock time, noting that wall-clock is lower-is-better and old `rmc-minimal` ran multiple internal reps, or use the old-format regex `steps/sec=([\d.]+)` with care because old output contained multiple per-rep matches.

Criterion micro-benchmarks remain useful for statistical comparisons of `rmc-core` hot-path changes. Use `make bench-before` to save a baseline and `make bench-after` to compare against it.
