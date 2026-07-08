//! Minimal 1-D random-walk simulation on the unified state-generic `rmc-core` API.
//!
//! The walker's integer position is the chain `State`. Each MC step proposes a `+/-1` move (always
//! accepted), and the per-cycle measurement records the current position. Because the position lives
//! in `State` — threaded by the run loop and owned independently per chain — there is no
//! `Arc<AtomicI64>` or other shared-mutable-state in the hot path, even for the parallel run.

use rmc::mc::{
    run_chain, Measurement, MetropolisKernel, NoopCallbacks, Runner, SimulationParams,
    SingleUpdateSet, Update,
};
use rmc::random::{ChainId, Rng, SeedSource};
use rmc::Merge;

#[derive(Clone)]
struct RandomWalkUpdate {
    proposed_delta: i64,
}

impl RandomWalkUpdate {
    fn new() -> Self {
        Self { proposed_delta: 0 }
    }
}

impl Update<i64> for RandomWalkUpdate {
    fn attempt<R: Rng + ?Sized>(&mut self, _position: &mut i64, rng: &mut R) -> f64 {
        self.proposed_delta = if rng.next_u64() & 1 == 0 { -1 } else { 1 };
        1.0
    }

    fn accept(&mut self, position: &mut i64) {
        *position += self.proposed_delta;
    }
}

struct WalkMeasurement {
    cycles: u64,
    final_position: i64,
}

impl WalkMeasurement {
    fn new() -> Self {
        Self {
            cycles: 0,
            final_position: 0,
        }
    }
}

impl Measurement<i64> for WalkMeasurement {
    type Output = WalkSummary;

    fn measure(&mut self, position: &i64) {
        self.cycles += 1;
        self.final_position = *position;
    }

    fn finish(self) -> Self::Output {
        WalkSummary {
            final_position_sum: self.final_position,
            measured_cycles: self.cycles,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct WalkSummary {
    final_position_sum: i64,
    measured_cycles: u64,
}

impl Merge for WalkSummary {
    fn merge(self, other: Self) -> Self {
        Self {
            final_position_sum: self.final_position_sum + other.final_position_sum,
            measured_cycles: self.measured_cycles + other.measured_cycles,
        }
    }
}

fn build_chain(
    _chain: ChainId,
) -> (
    i64,
    MetropolisKernel<SingleUpdateSet<RandomWalkUpdate>>,
    WalkMeasurement,
) {
    let state = 0_i64;
    let kernel = MetropolisKernel::new(SingleUpdateSet::new(RandomWalkUpdate::new()));
    (state, kernel, WalkMeasurement::new())
}

fn params() -> SimulationParams {
    SimulationParams {
        max_steps: 1_000,
        steps_per_cycle: 10,
        cycles_per_check: 1,
    }
}

fn main() -> rmc::Result<()> {
    let seed = SeedSource::new(0x5eed);
    let mut rng = seed.rng_for(ChainId(0));
    let (state, mut kernel, measurement) = build_chain(ChainId(0));
    let (_final_state, single_stats, single_summary) = run_chain(
        state,
        &mut rng,
        &mut kernel,
        measurement,
        params(),
        NoopCallbacks,
    )?;

    println!(
        "single chain: steps={}, cycles={}, final_position={}",
        single_stats.steps_done, single_stats.cycles_done, single_summary.final_position_sum
    );

    let chains = 8;
    let report = Runner::new(seed, build_chain)
        .chains(chains)
        .run(params())?;

    println!(
        "parallel: chains={}, steps={}, cycles={}, mean_final_position={:.3}",
        chains,
        report.stats.steps_done,
        report.stats.cycles_done,
        report.output.final_position_sum as f64 / chains as f64
    );

    Ok(())
}
