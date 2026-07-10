# plan-gpu.md — CubeCL GPU backend for IronMC

Goal: run many independent Fröhlich-polaron DiagMC chains on GPU (NVIDIA A100 via CUDA,
AMD 7900XTX via HIP/ROCm) with f64 precision, validated against the existing CPU engine.

## Decisions already made (do not re-litigate)

- **Stack**: CubeCL (pin the latest 0.10.x; verify current release at execution time —
  0.10.0 as of 2026-05). Kernels written in CubeCL's `#[cube]` Rust dialect, one source
  for CUDA + HIP backends. No wgpu backend in v1 (it is f32-only in practice).
- **Precision**: f64 everywhere in v1. A100 runs f64 at 1/2 rate (production); 7900XTX at
  1/32 rate (dev/correctness box — acceptable). Mixed precision is backlog, not v1.
- **Parallelism model**: one chain per GPU thread, thousands of chains. Warp-uniform
  update selection: all chains in a workgroup attempt the *same* update type each step
  (valid because update-type selection in `WeightedUpdateSet` is state-independent).
- **RNG**: Philox4x32-10, counter-based, keyed by `(run_seed, chain_id, step_counter)`.
  No per-thread mutable RNG state. Same generator used in CPU reference rigs so streams
  match across devices.
- **Order of work**: hardcoded fused Fröhlich kernel first (inside `rmc-frohlich`),
  generic framework driver (`rmc-cube`) extracted *afterwards* from the working instance.
- **State layout**: fixed-capacity flat SoA arrays per chain (no slotmap on GPU). Mirror
  the existing dense `storage`/`storage_idx` scheme from `crates/apps/rmc-frohlich/src/diagram.rs`
  with `u32` indices instead of `VKey`s.

## Global constraints (apply to every phase)

- Prefix all shell commands with `rtk` (e.g. `rtk cargo test --workspace`). See AGENTS.md.
- **Never run benchmarks yourself.** When benchmark verification is needed, print the
  exact command for the user and wait for their pasted results. Same for any GPU
  execution the local machine cannot do — mark those steps `USER CHECKPOINT`.
- `rtk cargo test --workspace` must pass at the end of every phase.
- Do not add GPU dependencies to `rmc-core` or the other framework crates — they stay
  dependency-light. CubeCL goes only into `rmc-frohlich` (optional, feature-gated) and
  later the new `rmc-cube` crate.
- Do not change the behavior of the existing CPU engine. All new code is additive; the
  only allowed touch to existing files is extracting pure functions (Phase 1) with
  `refactor:` commits. Flag any hot-path refactor for a user-run `bench-compare`.
- Commit subjects: `feat:` / `fix:` / `refactor:` / `chore:`, one phase step per commit.
- `crates/apps/rmc-frohlich/input.json` has uncommitted local changes — leave it alone.

---

## Phase 0 — Spike: validate CubeCL assumptions

Purpose: fail fast on the load-bearing assumptions before writing real code.
Create a scratch crate `crates/apps/cube-spike` (workspace member; deleted in Phase 5).

Environment check first: ROCm present? (`rtk which hipcc`, `ls /opt/rocm`). This is a
NixOS machine — if ROCm is missing, ask the user to provide a shell with it (e.g. a nix
shell with `rocmPackages.clr`) rather than installing anything yourself. If no GPU
runtime is available locally (e.g. an Intel-iGPU-only dev machine), run the spike on
CubeCL's `cpu` runtime instead — the MLIR/LLVM backend executes the same `#[cube]`
kernels on the host CPU, so the correctness items (1, 2, 3, 5) can be verified with no
GPU at all. Item 4 (atomics support) and anything performance-related still needs real
hardware and becomes a `USER CHECKPOINT` (compile-check locally, user runs and pastes
output).

Spike kernels (each ~20 lines, all f64):

1. **f64 + exp()**: kernel computing `exp(-lambda * dtau)` over an array; compare against
   CPU `f64::exp` — assert relative error `< 1e-14` elementwise. Run on HIP (XTX).
2. **Cube trait composition**: define a `#[cube]`-implemented trait (one method computing
   a weight ratio), call it from a generic driver kernel `fn drive<M: Model>(...)`. This
   validates the Phase 5 extraction path. If CubeCL 0.10 cannot express this, record the
   failure mode in this file and continue — Phase 3 does not depend on it.
3. **Runtime-bounded loop + divergent branches**: per-thread loop with an iteration count
   from a kernel arg, `if/else` on per-thread data. Sanity, not perf.
4. **Atomic f64 add** into a global buffer on both backends. If unsupported on either,
   record it: Phase 3 then uses the compare-and-swap fallback or u64 fixed-point atomics
   for histogram accumulation.
5. **Philox4x32-10** as a `#[cube]` fn *and* a plain Rust fn sharing the same constants.
   First check whether CubeCL 0.10 ships an RNG module (`cubecl-random` or in
   `cubecl-std`) — if yes and it is counter-based, use it and skip the hand-rolled one.
   Otherwise implement (~50 lines) and test both implementations against the published
   Random123 test vectors, plus CPU-vs-GPU agreement on 10^6 draws.

Acceptance: items 1, 3, 5 pass on at least the HIP backend; items 2 and 4 have a
recorded yes/no. Write results as a short table appended to this file under
`## Spike results`.

## Spike results

Environment checked 2026-07-10: `cubecl = 0.10.0` is the current 0.10.x release.
`rtk cargo check -p cube-spike --features cubecl-hip` and `--features cubecl-cuda`
both compile. This laptop has only an Intel Arrow Lake-U integrated GPU; `hipcc` and
`/opt/rocm` are absent, so HIP/CUDA execution remains a USER CHECKPOINT. Local runnable
checks use the CubeCL CPU runtime.

| Item | Local result | Device checkpoint |
|------|--------------|-------------------|
| f64 + `exp()` | PASS on CubeCL CPU; matches CPU `f64::exp` at `1e-14` rel. | Run on HIP. |
| Cube trait composition | PASS on CubeCL CPU with `#[cube] trait RatioModel` and generic driver kernel. | Run on HIP/CUDA if trait path is used in Phase 5. |
| Runtime loop + divergent branches | PASS on CubeCL CPU. | Run on HIP. |
| Atomic f64 add | CPU runtime reports unsupported (`None`); no kernel launch. | Run on HIP/CUDA; use CAS or fixed-point fallback if unsupported. |
| Philox4x32-10 | PASS: CPU implementation matches Random123 KATs; CubeCL CPU kernel matches CPU for 10^6 generated words. | Run the same check on HIP/CUDA. |

USER CHECKPOINT command on an AMD ROCm machine:

```bash
rtk cargo run -p cube-spike --features cubecl-hip
```

USER CHECKPOINT command on an NVIDIA CUDA machine:

```bash
rtk cargo run -p cube-spike --features cubecl-cuda
```

---

## Phase 1 — Flat SoA diagram engine on CPU (inside rmc-frohlich)

Purpose: the GPU-shaped data model, fully debuggable on CPU, validated against the
slotmap engine. This is the hard physics-touching phase.

1. **Extract pure scalar physics** from `diagram.rs` and `updates/` into a new
   `crates/apps/rmc-frohlich/src/physics.rs`: dispersion, phonon propagator, vertex
   factor, per-segment exponent `(dispersion(p) - mu) * dtau`, the acceptance-ratio
   formulas of all eight updates as functions of plain scalars/`[f64; 3]` (no `Diagram`,
   no `nalgebra` types — these become `#[cube]` leaves in Phase 3, and nalgebra will not
   cross that boundary). Existing engine calls the extracted functions; behavior
   identical. `refactor:` commit; ask the user to run the frohlich `bench-compare`
   command from README.md before merging this step (hot path).
2. **`FlatDiagram`** in new module `crates/apps/rmc-frohlich/src/flat/mod.rs`:
   - SoA fields: `tau: Vec<f64>`, `p_out: Vec<[f64; 3]>`, `q: Vec<[f64; 3]>`,
     `link/prev/next/storage_idx: Vec<u32>`, `phonons_above: Vec<u32>`, plus
     `storage: Vec<u32>`, `order`, and the same scalar params as `Diagram`.
   - Fixed capacity `2 * max_order_gpu + 2` slots, `u32::MAX` as the null index.
     `max_order_gpu` is a new `RunConfig` field (`#[serde(default = ...)]`, default 256).
     A capacity-exceeding proposal is an *impossible* move (return `< 0.0` acceptance,
     matching the existing `max_order` rejection semantics — same bounded model, so CPU
     validation runs must set the same `max_order`).
   - Free slots via swap-with-last on the dense `storage` list, mirroring the existing
     `storage_idx` bookkeeping.
3. **The eight updates against `FlatDiagram`** in `flat/updates.rs`, implementing the
   existing `rmc_core::mc::Update<FlatDiagram>` trait, calling only `physics.rs`
   functions for weights, and consuming RNG draws in *exactly the same order and count*
   as the slotmap updates (this enables the lockstep test).
4. **Validation** (all in `crates/apps/rmc-frohlich/tests/`):
   - `flat_ratio_parity.rs`: proptest — random valid configurations built in both
     representations, each update's acceptance probability equal to within `1e-12` rel.
   - `flat_lockstep.rs`: drive slotmap and flat engines from clones of the same seeded
     `DefaultRng` for 10^5 steps at defaults (α=1); assert identical accept/reject
     sequence and final `order`, `tau`, energy estimator within fp tolerance.
   - `flat_estimators.rs`: full short run through the existing `Runner` with the flat
     state; ground-state energy and self-energy histogram agree with the slotmap engine
     within 2σ jackknife errors.

Acceptance: all three tests green in `rtk cargo test -p rmc-frohlich`; workspace green.

`USER CHECKPOINT` (informational): flat vs slotmap `bench-compare` — flat likely wins on
CPU too; if it regresses, note it and continue (GPU is the target, not CPU perf).

---

## Phase 2 — Batched-chain driver semantics on CPU

Purpose: a CPU rig with *exactly* the GPU kernel's semantics: K chains in lockstep,
warp-uniform update selection, Philox streams. This is the executable spec for Phase 3
and stays as a test fixture.

1. `flat/batched.rs`: `run_batched(cfg, n_chains, group_size)` — flat chains advanced
   step-by-step; per step, update *type* drawn once per group of `group_size` chains
   from Philox keyed `(seed, group_id, step)`; per-chain proposal randomness keyed
   `(seed, chain_id, step, draw_index)`. Measurement every `steps_per_cycle` steps into
   per-chain accumulators.
2. **Jackknife over chain groups**: on GPU, batches = groups of chains (independent
   chains make cleaner jackknife batches than time batches). Map `n_batches` (256) to
   chain groups; reuse `BinnedBatchedSums` on the host by feeding per-group partial
   histograms. Per-group histogram memory: `num_bins × n_batches × 8 B` = 4 MB at
   defaults — this is the device-buffer plan for Phase 3.
3. **Self-consistent reweighting**: `energy_estimate` is re-fit between *segments*
   (a segment = `initial_self_consistent_period` steps, growing by `period_multiplier`),
   uniformly for all chains — mirroring what the host will do between GPU launches.
4. Test `batched_estimators.rs`: batched run (e.g. 64 chains × 10^6 steps, α=1) agrees
   with a Phase-1 single-chain run within 2σ. This also validates that warp-uniform
   update selection is unbiased.

Acceptance: test green; document in `flat/batched.rs` module docs that this rig is the
reference semantics for the GPU kernel (a `ponytail:` comment is fine).

---

## Phase 3 — Hardcoded Fröhlich CubeCL kernel

Purpose: the fused, model-specific GPU engine, feature-gated inside `rmc-frohlich`.
No framework abstraction yet.

1. **Cargo wiring**: optional deps `cubecl = { version = "0.10", optional = true }` with
   feature `gpu`, plus `gpu-cuda`, `gpu-hip`, and `gpu-cpu` features selecting the
   runtime (`gpu-cpu` = CubeCL's MLIR CPU backend, for GPU-less dev machines and CI).
   Nothing compiles differently without these features. Pin exact versions in the
   lockfile.
2. **Port physics leaves**: annotate/duplicate the `physics.rs` functions as `#[cube]`
   fns (f64) in `crates/apps/rmc-frohlich/src/gpu/physics.rs`. If `#[cube]` functions
   remain callable as plain Rust (spike will have shown this), share one definition with
   Phase 1 instead of duplicating; otherwise keep the pair adjacent with a unit test
   asserting CPU-vs-GPU agreement per function.
3. **State buffers**: SoA global-memory arrays `field[chain * capacity + slot]`,
   allocated from host; `u32` index arrays as in `FlatDiagram`. Upload initial states
   built by the Phase-1 CPU code (start from `start_tau`, order 0 — same as CPU).
4. **Driver kernel** `gpu/kernel.rs`: each thread owns one chain; loop over
   `steps_per_launch`; workgroup-uniform update-type draw (Philox keyed by group);
   attempt/accept inline (eight-way branch — divergence within the update's *proposal*
   randomness is expected and fine, the update *type* is uniform per workgroup);
   measurement every `steps_per_cycle` steps: scalar sums into per-chain f64 slots,
   self-energy histogram adds into the per-group `num_bins × n_batches` buffer (atomic
   f64 if the spike said yes, else the recorded fallback).
5. **Host loop** `gpu/run.rs`: launch in segments matching the self-consistent schedule;
   between segments download accumulators, re-fit `energy_estimate` exactly as
   `app.rs`/`measurement.rs` do, pass the new value as a kernel arg; final reduction
   feeds the existing `PolaronStats`/jackknife path so `RunOutput`, `write_results`, and
   `ValidationSummary` are reused unchanged.
6. **Update statistics**: per-update accept/propose/impossible counters (u32 per chain or
   atomics per group) surfaced through the existing `update_stats.rs` table, including
   the capacity-overflow rejection count (must be ~0 at validation parameters — if not,
   raise `max_order_gpu`).
7. **CLI**: `rmc-frohlich gpu <config.json> [results_dir]` in `main.rs`, gated by the
   feature; errors with a clear message when compiled without `gpu`.
8. **Tests**: everything not requiring a device stays in ordinary tests (stream keying,
   buffer layout round-trip, host reduction). Kernel-executing tests run on the CubeCL
   `cpu` runtime by default (`rtk cargo test -p rmc-frohlich --features gpu,gpu-cpu`) so
   they pass on GPU-less machines; the same tests against real GPUs run behind
   `#[ignore = "needs gpu"]` via
   `rtk cargo test -p rmc-frohlich --features gpu,gpu-hip -- --ignored`.

`USER CHECKPOINT` — validation runs on the 7900XTX (user executes, pastes output):
- Short trajectory check: 64 chains × 10^4 steps, GPU vs Phase-2 CPU rig with identical
  Philox streams. Accept/reject sequences will eventually diverge from `exp()` ULP
  differences between libm and the GPU — compare only the first divergence-free prefix
  and require it to be long (>10^3 steps); a divergence at step ~1 means a real bug.
- Statistical check: `chains ≥ 4096`, α=1, defaults otherwise; E₀ and self-energy
  histogram within 2σ of a CPU run of comparable statistics.

Acceptance: both checks pass on HIP; workspace tests green with and without `gpu`.

---

## Phase 4 — A100 / CUDA validation

`USER CHECKPOINT` throughout (cluster access is the user's).

1. User builds with `--features gpu,gpu-cuda` on the cluster and repeats the Phase-3
   statistical check on an A100. Same numbers expected (up to RNG-identical: the Philox
   streams match, so results should be *very* close, not just within 2σ).
2. Cross-backend comparison: XTX vs A100 outputs at identical config + seed.
3. Throughput survey (informational, no target): steps/sec at chains ∈
   {1k, 4k, 16k, 64k} vs the CPU `bench` number. Record in this file.

Acceptance: CUDA results consistent with HIP and CPU. Record numbers under
`## Validation results`.

---

## Phase 5 — Extract the generic driver into `crates/rmc-cube`

Purpose: turn the working hardcoded kernel into the framework API. Only start after
Phase 3 is validated; the shape of the trait comes from what the frohlich kernel
actually needed, plus the spike's answer on cube-trait generics.

1. New crate `crates/rmc-cube` (workspace member, depends on `cubecl` + `rmc-core`):
   - `Model` cube trait: comptime state stride/capacity, `init`, per-update
     `attempt/accept` leaves, `measure`. Exact shape = whatever Phase 3's kernel calls.
   - Generic driver kernel, Philox module, launch/segment host loop, accumulator
     download + `Merge` integration, backend selection.
   - If the spike showed cube-trait generics don't work: fall back to a macro that
     stamps the driver out per model (`dispatch_update!` precedent exists in rmc-core).
2. Port `rmc-frohlich`'s GPU path onto `rmc-cube`; delete the now-redundant driver code
   from `gpu/`; the physics leaves stay in the app.
3. Regression: identical config + seed produces identical output before/after the
   extraction (same Philox streams — this must be exact, not statistical).
4. Delete `crates/apps/cube-spike`. Update AGENTS.md repo map and README architecture
   section (one paragraph: `rmc-cube` = GPU driver; apps supply `#[cube]` physics
   leaves). `chore:` commit.

Acceptance: regression exact-match passes (user runs the GPU halves); workspace green;
docs updated.

---

## Backlog (explicitly out of scope for v1)

- Warm-chain checkpointing (serialize device state, resume without re-thermalizing).
- Mixed-precision experiment (f32 proposals + f64 exponent accumulation), preceded by
  the CPU f32-vs-f64 bias study.
- wgpu/f32 backend for portability.
- Derive macro for SoA state layout (add when a second GPU app exists).
- Multi-GPU / multi-node.

## Risks & fallbacks

| Risk                                      | Detection          | Fallback                                                                                   |
|-------------------------------------------|--------------------|--------------------------------------------------------------------------------------------|
| Cube-trait generics can't express `Model` | Spike item 2       | Macro-stamped driver (Phase 5.1); Phase 3 unaffected                                       |
| No f64 atomics on a backend               | Spike item 4       | CAS-loop or u64 fixed-point atomics for histograms                                         |
| No ROCm on dev box                        | Phase 0 env check  | User provides nix shell; until then compile-check + user-run checkpoints                   |
| CubeCL 0.10 bug/limitation blocks kernel  | Phase 3            | Report upstream; worst case: HIP C++ single-source kernels via FFI, keep everything else   |
| Lockstep CPU/GPU diverges immediately     | Phase 3 checkpoint | Bug in stream keying or physics port — bisect per-update with ratio-parity tests on device |
| f64 perf on XTX disappointing             | Phase 4 survey     | Expected (1/32 rate); production is the A100. Mixed precision stays backlog                |
