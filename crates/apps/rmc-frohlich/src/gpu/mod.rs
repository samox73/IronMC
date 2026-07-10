//! Feature-gated Phase-3 GPU entry points.
//!
//! The local `gpu-cpu` path intentionally reuses the Phase-2 batched CPU rig. It is the executable
//! spec for stream keying, group-uniform update selection, state layout, and host reduction while
//! HIP/CUDA execution stays behind the user checkpoints in `plan-gpu.md`.

pub mod kernel;
pub mod physics;
pub mod run;
pub mod state;

pub use run::run_gpu_from_config;
