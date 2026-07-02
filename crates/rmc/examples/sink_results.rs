//! Result-sink measurement example.
//!
//! The measurement path is output-only: results are written as named artifacts into a sink instead
//! of returned as Rust-native typed values.

use rmc::io::MapSink;
use rmc::mc::{
    run_with_sink, MetropolisKernel, ResultSink, SimulationParams, SingleUpdateSet,
    SinkMeasurement, SinkMeasurementSet, Update,
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
struct WalkSinkMeasurement {
    cycles: u64,
    final_position: i64,
}

impl SinkMeasurement<i64> for WalkSinkMeasurement {
    fn name(&self) -> &str {
        "walk"
    }

    fn measure(&mut self, position: &i64) {
        self.cycles += 1;
        self.final_position = *position;
    }

    fn write_result(&self, sink: &mut dyn ResultSink) -> rmc::Result<()> {
        sink.put("cycles", &self.cycles)?;
        sink.put("final_position", &self.final_position)
    }
}

fn main() -> rmc::Result<()> {
    let mut rng = SeedSource::new(0x51_ace).rng_for(ChainId(0));
    let mut kernel = MetropolisKernel::new(SingleUpdateSet::new(RandomWalkUpdate::new()));
    let mut measurements = SinkMeasurementSet::new();
    measurements.add(WalkSinkMeasurement::default())?;

    let mut sink = MapSink::new();
    let (_state, stats) = run_with_sink(
        0_i64,
        &mut rng,
        &mut kernel,
        &mut measurements,
        &mut sink,
        SimulationParams {
            max_steps: 1_000,
            steps_per_cycle: 10,
            cycles_per_check: 1,
        },
    )?;

    println!("steps={}, cycles={}", stats.steps_done, stats.cycles_done);
    for (path, value) in sink.results() {
        println!("{path}={value}");
    }

    Ok(())
}
