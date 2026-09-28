# Example applications

Complete programs built on the IronMC framework. Each one shows how a real simulation plugs its own state, updates, and measurements into the `rmc-core` run loop, and doubles as a fixture for the framework's performance-regression benchmarks (see the top-level README).

| App | What it is | Docs |
|---|---|---|
| `rmc-minimal` | 1-D Gaussian sampler with three toy updates; the framework hot-path benchmark | [README](rmc-minimal/README.md) |
| `rmc-frohlich` | Diagrammatic Monte Carlo for the Fröhlich polaron self-energy, CPU and CubeCL GPU backends; the full-application benchmark | [README](rmc-frohlich/README.md) |
| `cube-spike` | CubeCL feasibility spike: checks the GPU kernel features the GPU backends rely on | [README](cube-spike/README.md) |

Quick start:

```sh
make run                                         # rmc-frohlich against its input.json (see the Makefile)
cargo run --release -p rmc-minimal -- full 1000000
cargo run --release -p cube-spike --features cubecl-cpu
```
