//! Typed measurements composed as a tuple, then named with plain JSON.

use rmc::mc::{
    run_chain, Measurement, MetropolisKernel, NoopCallbacks, SimulationParams, SingleUpdateSet,
    Update,
};
use rmc::random::{ChainId, Rng, SeedSource};

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

#[derive(Default)]
struct SampleCount(u64);

impl Measurement<i64> for SampleCount {
    type Output = u64;

    fn measure(&mut self, _position: &i64) {
        self.0 += 1;
    }

    fn finish(self) -> Self::Output {
        self.0
    }
}

#[derive(Default)]
struct FinalPosition(i64);

impl Measurement<i64> for FinalPosition {
    type Output = i64;

    fn measure(&mut self, position: &i64) {
        self.0 = *position;
    }

    fn finish(self) -> Self::Output {
        self.0
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rng = SeedSource::new(0x51_ace).rng_for(ChainId(0));
    let mut kernel = MetropolisKernel::new(SingleUpdateSet::new(RandomWalkUpdate::new()));
    let (_state, stats, (samples, final_position)) = run_chain(
        0_i64,
        &mut rng,
        &mut kernel,
        (SampleCount::default(), FinalPosition::default()),
        SimulationParams {
            max_steps: 1_000,
            steps_per_cycle: 10,
            cycles_per_check: 1,
        },
        NoopCallbacks,
    )?;

    let results = serde_json::json!({
        "stats/steps": stats.steps_done,
        "stats/cycles": stats.cycles_done,
        "walk/samples": samples,
        "walk/final_position": final_position,
    });
    println!("{}", serde_json::to_string_pretty(&results)?);

    Ok(())
}
