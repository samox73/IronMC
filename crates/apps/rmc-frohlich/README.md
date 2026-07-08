# rmc-frohlich

```mermaid
graph TD
    cfg["config.rs<br/>RunConfig"] --> app["app.rs<br/>run_from_config_with_progress / run_bench"]

    subgraph plugs["plugged into the framework"]
        diag["diagram.rs<br/>Diagram / Vertex state"]
        upd["updates/<br/>PolaronUpdate moves"]
        meas["measurement.rs<br/>PolaronMeasurement estimators"]
    end

    app --> diag
    app --> upd
    app --> meas
    diag --> loop(["rmc run loop"])
    upd --> loop
    meas --> loop

    loop --> app
    app --> fourier["fourier.rs<br/>post-processing"]
    app --> sanity["sanity.rs<br/>checks"]
    fourier --> results["write_results → results/*.json"]
    sanity --> results
```

Diagrammatic Monte Carlo for the Fröhlich polaron self-energy. The "real application" fixture
for this workspace, as opposed to the toy `rmc-minimal` benchmark.

## Run

```bash
make run                                     # release build against ./input.json
cargo run -p rmc-frohlich -- def             # print the default RunConfig as JSON
cargo run -p rmc-frohlich -- bench           # timed sampling loop only, no output files
cargo run -p rmc-frohlich -- <config.json> [results_dir]   # full run with progress bar
```

Results (config, summary, raw stats, self-energy, FFT, checkpoint) are written as JSON to
`results/` by default.
