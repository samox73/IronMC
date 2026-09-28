# cube-spike

A CubeCL feasibility spike, not a physics application. It runs small kernels that exercise the GPU features the CubeCL backends depend on, and checks each against a host reference:

- `exp` accuracy in f64 (relative error ≤ 1e-14),
- trait composition inside `#[cube]` functions,
- data-dependent loops and branches,
- f64 atomic add,
- the counter-based Philox4x32-10 RNG, bit-exact against the host implementation over 10⁶ words.

## Run

Pick one runtime feature; without one the binary only prints a hint.

```sh
cargo run --release -p cube-spike --features cubecl-cpu    # CubeCL CPU runtime, no GPU needed
cargo run --release -p cube-spike --features cubecl-cuda   # NVIDIA
cargo run --release -p cube-spike --features cubecl-hip    # AMD
```

Each check prints `pass` (or the atomic result) and the binary panics on the first mismatch.
