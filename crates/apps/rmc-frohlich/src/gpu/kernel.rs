use crate::config::RunConfig;
use crate::flat::batched::{run_batched, BatchedRunOutput};

pub const DEFAULT_WORKGROUP_SIZE: usize = 64;

pub fn launch_reference_kernel(cfg: &RunConfig) -> rmc_core::Result<BatchedRunOutput> {
    run_batched(cfg, cfg.chains as usize, DEFAULT_WORKGROUP_SIZE)
}
