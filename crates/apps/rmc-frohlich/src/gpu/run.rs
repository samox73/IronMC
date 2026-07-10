use crate::app::{AppResult, RunOutput};
use crate::config::RunConfig;
use crate::gpu::kernel::launch_reference_kernel;

pub fn run_gpu_from_config(cfg: &RunConfig) -> AppResult<RunOutput> {
    let start = std::time::Instant::now();
    let output = launch_reference_kernel(cfg)?;
    Ok(RunOutput {
        stats: output.stats,
        measurement: output.measurement,
        final_state: None,
        update_stats: output.update_stats,
        wall_secs: start.elapsed().as_secs_f64(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gpu_cpu_reference_smoke() {
        let cfg = RunConfig {
            chains: 4,
            max_steps: 20,
            steps_per_cycle: 5,
            n_batches: 4,
            num_bins: 20,
            initial_self_consistent_period: usize::MAX,
            ..RunConfig::default()
        };
        let output = run_gpu_from_config(&cfg).unwrap();
        assert_eq!(output.stats.steps_done, 80);
        assert_eq!(output.measurement.sample_count, 16);
        assert!(output.final_state.is_none());
        assert_eq!(
            output
                .update_stats
                .iter()
                .map(|row| row.proposed)
                .sum::<u64>(),
            80
        );
    }
}
